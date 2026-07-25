from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.state import State
from app.prompts import SYSTEM_PROMPT
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import os
import sys
from app.utils.approval.approval_tools import APPROVAL_REQUIRED_TOOLS
from langchain_core.tools import StructuredTool
from app.utils.approval.approval import require_approval
import json
from app.llms.openrouter_llm import llm

graph = None
mcp_client = None


def wrap_tool_with_approval(tool):

    if tool.name not in APPROVAL_REQUIRED_TOOLS:
        return tool

    async def execute_with_approval(**kwargs):

        user_response = require_approval(
            tool_name=tool.name,
            data=kwargs,
            message=APPROVAL_REQUIRED_TOOLS[tool.name]["message"],
        )

        if user_response.lower().strip() in [
            "send",
            "yes",
            "approve",
            "confirm",
            "ok",
        ]:
            return await tool.ainvoke(kwargs)

        if user_response.lower().strip() in [
            "cancel",
            "no",
            "reject",
            "stop",
        ]:
            return {
                "success": False,
                "status": "cancelled",
                "message": f"{tool.name} was cancelled by the user.",
            }

        return {
            "success": False,
            "status": "change_requested",
            "tool": tool.name,
            "current_data": kwargs,
            "user_request": user_response,
            "message": (
                "The user does not want to execute the tool yet. "
                "Update the tool arguments according to user_request "
                "and call the tool again for approval."
            ),
        }

    return StructuredTool.from_function(
        coroutine=execute_with_approval,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def check_cancelled(state: State):
    """
    Runs right after the 'tools' node.
    If the last tool result was a cancellation, skip the LLM entirely
    and go straight to a fixed 'cancelled' response — instead of letting
    the LLM decide what to do (which was causing retries / silent bypass).
    """
    last_msg = state["messages"][-1]

    if isinstance(last_msg, ToolMessage):
        content = last_msg.content
        try:
            data = content if isinstance(content, dict) else json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = {}

        if isinstance(data, dict) and data.get("status") == "cancelled":
            return "end_cancelled"

    return "chatbot"


def end_cancelled(state: State):
    last_msg = state["messages"][-1]
    tool_name = "that action"
    try:
        content = last_msg.content
        data = content if isinstance(content, dict) else json.loads(content)
        tool_name = data.get("message", tool_name)
    except Exception:
        pass

    return {
        "messages": [
            AIMessage(
                content=f"Okay, I cancelled that action. Let me know what you'd like to do instead."
            )
        ]
    }


async def init_graph():
    global graph, mcp_client
    mcp_client = MultiServerMCPClient(
        {
            "assistant": {
                "command": sys.executable,
                "args": ["-m", "app.mcp_server"],
                "transport": "stdio",
                "cwd": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()
    tools = [wrap_tool_with_approval(tool) for tool in mcp_tools]
    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state: State):
        res = await llm_with_tools.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [res]}

    checkpointer = InMemorySaver()

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("end_cancelled", end_cancelled)

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    builder.add_conditional_edges(
        "tools",
        check_cancelled,
        {
            "end_cancelled": "end_cancelled",
            "chatbot": "chatbot",
        },
    )
    builder.add_edge("end_cancelled", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph
