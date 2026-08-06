"""
The MCP tools (Gmail, Calendar, Meet) run in a separate stdio subprocess
(app/mcp_server.py) and have no visibility into the current HTTP request or
which user is chatting. This interceptor bridges that gap: it reads the
authenticated user_id from the graph's runtime context and silently injects
it as a tool argument before the call reaches the subprocess.

The LLM never sees `user_id` in the tool schema — it's added here, after
the model has already decided which tool to call and with what arguments.
"""

from langchain_mcp_adapters.interceptors import MCPToolCallRequest

# Tools that need to know which user's Google account to act on.
GOOGLE_TOOLS = {
    "get_recent_emails",
    "search_emails",
    "get_email",
    "send_email",
    "create_draft",
    "mark_email_read",
    "mark_email_unread",
    "list_email_attachments",
    "download_email_attachment",
    "get_upcoming_events",
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    "create_google_meet",
    "get_google_meet",
    "end_google_meet",
}


async def inject_user_context(request: MCPToolCallRequest, handler):
    if request.name in GOOGLE_TOOLS:
        runtime = request.runtime
        user_id = getattr(getattr(runtime, "context", None), "user_id", None)

        if not user_id:
            raise RuntimeError(
                f"Cannot call '{request.name}': no authenticated user_id in context."
            )

        request = request.override(args={**request.args, "user_id": user_id})

    return await handler(request)
