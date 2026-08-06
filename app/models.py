from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    attached_files: list[str] | None = None
