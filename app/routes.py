from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app import graph as graph_module
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage, ToolMessage
from app.utils.format_messages import format_messages
from langgraph.types import Command
from fastapi.responses import StreamingResponse
import json

router = APIRouter()


def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_graph_response(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state = graph_module.graph.get_state(config)

    resuming = bool(state.tasks and any(task.interrupts for task in state.tasks))

    if resuming:
        input_ = Command(resume=request.message)
    else:
        input_ = {"messages": [HumanMessage(content=request.message)]}

    streamed_any = False  # track whether any LLM token actually streamed

    # track tool calls we've already announced, keyed by tool_call id,
    # so we don't spam "tool_start" once per chunk while args stream in
    announced_tool_calls = set()

    try:
        async for msg_chunk, metadata in graph_module.graph.astream(
            input_,
            config=config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node")

            # --- 1. Tool call being requested by the model ---
            if isinstance(msg_chunk, AIMessageChunk):
                tool_call_chunks = getattr(msg_chunk, "tool_call_chunks", None) or []
                for tc in tool_call_chunks:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name")
                    if tc_name and tc_id and tc_id not in announced_tool_calls:
                        announced_tool_calls.add(tc_id)
                        registry_entry = graph_module.TOOL_REGISTRY.get(tc_name, {})
                        yield sse(
                            "tool_start",
                            {
                                "tool_call_id": tc_id,
                                "tool_name": tc_name,
                                "tool_label": registry_entry.get("label", tc_name),
                                "node": node,
                            },
                        )

                # --- 2. Thinking / reasoning content (extended thinking models) ---
                # content can be a plain string OR a list of content blocks
                if isinstance(msg_chunk.content, list):
                    for block in msg_chunk.content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type == "thinking" and block.get("thinking"):
                            yield sse(
                                "thinking",
                                {
                                    "content": block["thinking"],
                                    "node": node,
                                },
                            )
                        elif block_type == "text" and block.get("text"):
                            streamed_any = True
                            yield sse(
                                "token",
                                {"content": block["text"], "node": node},
                            )
                elif msg_chunk.content:
                    streamed_any = True
                    yield sse(
                        "token",
                        {"content": msg_chunk.content, "node": node},
                    )

            # --- 3. Tool has finished executing and returned a result ---
            elif isinstance(msg_chunk, ToolMessage):
                yield sse(
                    "tool_end",
                    {
                        "tool_call_id": msg_chunk.tool_call_id,
                        "tool_name": msg_chunk.name,
                        "node": node,
                        "status": "error" if msg_chunk.status == "error" else "success",
                    },
                )

    except Exception as e:
        yield sse("error", {"message": str(e)})
        return

    final_state = graph_module.graph.get_state(config)

    # HITL interrupt check (unchanged)
    if final_state.tasks and any(task.interrupts for task in final_state.tasks):
        interrupt_data = final_state.tasks[0].interrupts[0].value
        yield sse(
            "approval_required",
            {
                "thread_id": request.thread_id,
                "status": "approval_required",
                "approval": interrupt_data,
            },
        )
        return

    if not streamed_any:
        last_msg = final_state.values["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            yield sse(
                "token",
                {
                    "content": last_msg.content,
                    "node": "fallback",
                },
            )

    messages = format_messages(final_state.values["messages"], request.thread_id)
    yield sse(
        "done",
        {
            "thread_id": request.thread_id,
            "status": "completed",
            "response": messages,
            "total_records": len(messages),
        },
    )


@router.post("/chat/stream")
async def streamChat(request: ChatRequest):
    return StreamingResponse(
        stream_graph_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(request: ChatRequest):

    config = {"configurable": {"thread_id": request.thread_id}}

    state = graph_module.graph.get_state(config)

    # Graph is currently waiting at interrupt()
    if state.tasks and any(task.interrupts for task in state.tasks):

        result = await graph_module.graph.ainvoke(
            Command(resume=request.message),
            config=config,
        )

    # Normal conversation
    else:

        result = await graph_module.graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

    # HITL triggered
    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        return {
            "thread_id": request.thread_id,
            "status": "approval_required",
            "approval": interrupt_data,
        }

    # Normal response
    state = graph_module.graph.get_state(config)

    messages = format_messages(state.values["messages"], request.thread_id)

    return {
        "thread_id": request.thread_id,
        "status": "completed",
        "response": messages,
        "total_records": len(messages),
    }


@router.get("/chat/{thread_id}")
async def allChats(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_module.graph.get_state(config)
    if not state.values:
        raise HTTPException(status_code=400, detail="User not found")

    message = format_messages(state.values["messages"], thread_id)

    return {
        "thread_id": thread_id,
        "response": message,
        "total_records": len(message),
    }
