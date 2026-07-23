from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.state import State
from app.prompts import SYSTEM_PROMPT
from app.config import OPENAI_API_KEY, OPENROUTER_API_KEY
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import os
import sys
from langgraph.types import interrupt
from app.utils.approval.approval_tools import APPROVAL_REQUIRED_TOOLS
from langchain_core.tools import StructuredTool
from app.utils.approval.approval import require_approval

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
)

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

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    builder.add_edge("tools", "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph
