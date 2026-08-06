from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from app.models import ChatRequest
from app import graph as graph_module
from app.state import Context
from app.auth.session import get_current_user_id, get_current_user_id_optional
from app.mcp_interceptors import LoginRequiredError
from app.db import get_user_by_id
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage, ToolMessage
from app.utils.format_messages import format_messages
from langgraph.types import Command
from fastapi.responses import StreamingResponse
import json
import shutil
from pathlib import Path
import os
import uuid

router = APIRouter()


def user_thread_id(user_id: str | None, guest_id: str | None = None) -> str:
    """
    Signed-in users get a stable per-account thread (`user-<id>`), so their
    history persists across visits. Guests get an ephemeral thread scoped
    to a random id generated per browser session (see /chat/stream) so
    concurrent anonymous visitors never share state, but nothing is kept
    long-term for them.
    """
    if user_id:
        return f"user-{user_id}"
    return f"guest-{guest_id}"


@router.get("/me")
async def get_me(user_id: str = Depends(get_current_user_id)):
    """Returns the signed-in user's basic profile, for the frontend to render the sidebar."""
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "id": user["_id"],
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture"),
    }


EXPORTS_BASE_DIR = Path("exports")
GUEST_COOKIE_NAME = "guest_id"
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".xls",
    ".csv",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".txt",
}
MAX_UPLOAD_SIZE = 15 * 1024 * 1024


def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_user_content(request: ChatRequest) -> str:
    content = request.message
    if request.attached_files:
        files_note = ", ".join(request.attached_files)
        content = f"{content}\n\n[Attached files available for use: {files_note}]"
    return content


async def stream_graph_response(
    request: ChatRequest, user_id: str | None, guest_id: str | None
):
    thread_id = user_thread_id(user_id, guest_id)
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(user_id=user_id)
    state = graph_module.graph.get_state(config)

    resuming = bool(state.tasks and any(task.interrupts for task in state.tasks))

    if resuming:
        input_ = Command(resume=request.message)
    else:
        input_ = {"messages": [HumanMessage(content=build_user_content(request))]}

    streamed_any = False  # track whether any LLM token actually streamed

    # track tool calls we've already announced, keyed by tool_call id,
    # so we don't spam "tool_start" once per chunk while args stream in
    announced_tool_calls = set()

    try:
        async for msg_chunk, metadata in graph_module.graph.astream(
            input_,
            config=config,
            context=context,
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
                is_error = msg_chunk.status == "error"
                content_str = (
                    msg_chunk.content
                    if isinstance(msg_chunk.content, str)
                    else str(msg_chunk.content)
                )

                if is_error and LoginRequiredError.MARKER in content_str:
                    yield sse(
                        "login_required",
                        {
                            "message": "Sign in with Google to use email, calendar, and meeting features.",
                        },
                    )
                    return

                yield sse(
                    "tool_end",
                    {
                        "tool_call_id": msg_chunk.tool_call_id,
                        "tool_name": msg_chunk.name,
                        "node": node,
                        "status": "error" if is_error else "success",
                    },
                )

    except LoginRequiredError:
        yield sse(
            "login_required",
            {
                "message": "Sign in with Google to use email, calendar, and meeting features.",
            },
        )
        return

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
                "thread_id": thread_id,
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

    messages = format_messages(final_state.values["messages"], thread_id)
    yield sse(
        "done",
        {
            "thread_id": thread_id,
            "status": "completed",
            "response": messages,
            "total_records": len(messages),
        },
    )


@router.post("/chat/stream")
async def streamChat(
    request: ChatRequest,
    http_request: Request,
    user_id: str | None = Depends(get_current_user_id_optional),
):
    guest_id = None
    set_guest_cookie = False

    if not user_id:
        guest_id = http_request.cookies.get(GUEST_COOKIE_NAME)
        if not guest_id:
            guest_id = uuid.uuid4().hex
            set_guest_cookie = True

    response = StreamingResponse(
        stream_graph_response(request, user_id, guest_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

    if set_guest_cookie:
        # Lets an anonymous visitor keep a consistent (but unlinked-to-any-
        # account) conversation thread across messages in the same browser,
        # without requiring sign-in. Cleared automatically when they log in
        # (their real user_id thread takes over).
        response.set_cookie(
            key=GUEST_COOKIE_NAME,
            value=guest_id,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=7 * 24 * 60 * 60,
        )

    return response


@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):

    thread_id = user_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(user_id=user_id)

    state = graph_module.graph.get_state(config)

    # Graph is currently waiting at interrupt()
    if state.tasks and any(task.interrupts for task in state.tasks):

        result = await graph_module.graph.ainvoke(
            Command(resume=request.message),
            config=config,
            context=context,
        )

    # Normal conversation
    else:

        result = await graph_module.graph.ainvoke(
            {"messages": [HumanMessage(content=build_user_content(request))]},
            config=config,
            context=context,
        )

    # HITL triggered
    if "__interrupt__" in result:

        interrupt_data = result["__interrupt__"][0].value

        return {
            "thread_id": thread_id,
            "status": "approval_required",
            "approval": interrupt_data,
        }

    # Normal response
    state = graph_module.graph.get_state(config)

    messages = format_messages(state.values["messages"], thread_id)

    return {
        "thread_id": thread_id,
        "status": "completed",
        "response": messages,
        "total_records": len(messages),
    }


@router.get("/chat")
async def allChats(user_id: str = Depends(get_current_user_id)):
    thread_id = user_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_module.graph.get_state(config)
    if not state.values:
        # No conversation yet for this user — return an empty history, not an error.
        return {"thread_id": thread_id, "response": [], "total_records": 0}

    message = format_messages(state.values["messages"], thread_id)

    return {
        "thread_id": thread_id,
        "response": message,
        "total_records": len(message),
    }


@router.delete("/chat")
async def clearChat(user_id: str = Depends(get_current_user_id)):
    thread_id = user_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}

    state = graph_module.graph.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        await graph_module.graph.checkpointer.adelete_thread(thread_id)
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Checkpointer does not support thread deletion; implement manual cleanup.",
        )

    # Only delete this user's own uploaded/exported files.
    user_exports_dir = EXPORTS_BASE_DIR / user_id
    deleted_files = 0
    if user_exports_dir.exists():
        for file_path in user_exports_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
                deleted_files += 1

    return {
        "thread_id": thread_id,
        "status": "cleared",
        "history_cleared": True,
        "files_deleted": deleted_files,
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    # Each user's uploads live in their own subfolder so filenames never collide
    # across accounts and one user can never read another user's files.
    user_dir = EXPORTS_BASE_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(file.filename)
    dest_path = user_dir / safe_name
    stem, suffix = os.path.splitext(safe_name)
    counter = 1
    while dest_path.exists():
        dest_path = user_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    size = 0
    with open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                out.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File too large (max 15MB)")
            out.write(chunk)

    return {
        "success": True,
        "file_name": dest_path.name,
        "size": size,
        "content_type": file.content_type,
    }


@router.get("/healthz")
async def health_check():
    return {"status": "ok"}
