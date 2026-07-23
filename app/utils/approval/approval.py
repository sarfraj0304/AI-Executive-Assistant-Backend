from langgraph.types import interrupt


def require_approval(
    tool_name: str,
    data: dict,
    message: str = "Do you want to continue?",
):
    """
    Generic HITL approval function.

    Returns the user's decision after LangGraph resumes.
    """

    return interrupt(
        {
            "tool": tool_name,
            "message": message,
            "data": data,
        }
    )
