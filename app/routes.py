from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app import graph as graph_module
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
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

    try:
        async for msg_chunk, metadata in graph_module.graph.astream(
            input_,
            config=config,
            stream_mode="messages",
        ):
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                streamed_any = True
                yield sse(
                    "token",
                    {
                        "content": msg_chunk.content,
                        "node": metadata.get("langgraph_node"),
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
            "X-Accel-Buffering": "no",  # disable nginx buffering if you're behind it
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
