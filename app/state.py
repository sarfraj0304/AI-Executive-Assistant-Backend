from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, BaseMessage


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
