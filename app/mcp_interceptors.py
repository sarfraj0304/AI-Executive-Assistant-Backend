from langchain_mcp_adapters.interceptors import MCPToolCallRequest


class LoginRequiredError(Exception):

    MARKER = "__LOGIN_REQUIRED__"

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            f"{self.MARKER}Cannot call '{tool_name}': no authenticated user. "
            f"Sign in with Google to use email, calendar, and meeting features."
        )


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
        context = getattr(runtime, "context", None)
        context_user_id = getattr(context, "user_id", None)
        user_id = context_user_id or request.args.get("user_id")

        if not user_id:
            raise LoginRequiredError(request.name)

        request = request.override(args={**request.args, "user_id": user_id})

    return await handler(request)
