from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, BaseMessage
from dataclasses import dataclass


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str | None


@dataclass
class Context:
    user_id: str | None = None
