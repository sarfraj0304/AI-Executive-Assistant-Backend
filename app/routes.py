from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app import graph as graph_module
from langchain_core.messages import HumanMessage
from app.utils.format_messages import format_messages
from langgraph.types import Command

router = APIRouter()


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
