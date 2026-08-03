from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    attached_files: list[str] | None = None
