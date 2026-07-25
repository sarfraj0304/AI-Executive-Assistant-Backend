import os
import uuid
from io import BytesIO
from pypdf import PdfReader
from langchain_core.messages import SystemMessage, HumanMessage
from app.resume_worker.resume import ResumeSchema
from app.llms.openai_llm import open_ai_llm

# Backend root directory (same convention as app/tools/export.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

RESUME_EXTRACTION_PROMPT = """
You are a resume-parsing engine.

Read the raw resume text below and extract the candidate's information
into the given structured schema.

Rules:
- Never invent information that isn't present in the text.
- If a field isn't present, leave it null / empty.
- Normalize dates where possible (e.g. "Jan 2022").
- Infer total_years_experience only if it can be reasonably
  calculated from the work history dates; otherwise leave it null.
"""


def save_uploaded_resume(file_bytes: bytes, original_name: str) -> str:
    """
    Save an uploaded resume PDF to the uploads dir.
    Returns the unique file_name the agent/tool will later reference.
    """

    unique_name = f"{uuid.uuid4().hex}_{original_name}"

    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return unique_name


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an in-memory PDF file."""

    reader = PdfReader(BytesIO(file_bytes))

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text_parts.append(page_text)

    text = "\n".join(text_parts).strip()

    if not text:
        raise ValueError(
            "No extractable text found in PDF. "
            "It may be a scanned/image-only resume that needs OCR."
        )

    return text


async def parse_resume_by_file_name(file_name: str) -> dict:
    """
    Full pipeline used by the MCP tool: file_name (already on disk in
    uploads/) -> raw text -> structured resume data (as a plain dict,
    since MCP tool results must be JSON-serializable).
    """

    # Prevent path traversal - same guard your send_email tool uses
    safe_name = os.path.basename(file_name)

    file_path = os.path.join(UPLOAD_DIR, safe_name)

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": "file_not_found",
            "message": f"No uploaded resume found with file_name: {safe_name}",
        }

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    try:
        raw_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        return {
            "success": False,
            "error": "no_extractable_text",
            "message": str(e),
        }

    structured_llm = open_ai_llm.with_structured_output(ResumeSchema)

    result = await structured_llm.ainvoke(
        [
            SystemMessage(content=RESUME_EXTRACTION_PROMPT),
            HumanMessage(content=raw_text),
        ]
    )

    return {
        "success": True,
        "file_name": safe_name,
        "data": result.model_dump() if hasattr(result, "model_dump") else result,
    }
