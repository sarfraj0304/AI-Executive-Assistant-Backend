from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, BaseMessage
from dataclasses import dataclass


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@dataclass
class Context:
    """
    Run-scoped context passed into graph.astream(..., context=Context(user_id=...)).
    Used by the MCP tool interceptor to inject user_id into Gmail/Calendar/Meet
    tool calls without the LLM ever seeing or choosing that argument.
    """

    user_id: str
