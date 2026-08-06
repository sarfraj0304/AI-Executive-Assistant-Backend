import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
from langchain_tavily import TavilySearch
from app.tools.gmail import get_gmail_service
import base64
from email.message import EmailMessage
from app.tools.calender import get_calendar_service
from datetime import datetime, timezone
from app.tools.export import (
    create_pdf,
    create_excel,
)
import mimetypes
import base64
from app.tools.meet import get_meet_service
from pypdf import PdfReader
from docx import Document

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("chatbot_mcp")

MAX_CHARS = 15000


@mcp.tool()
async def get_weather(city_name: str):
    """used for getting weather information of given city"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        data = res.json()

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }


@mcp.tool()
async def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """get the latest exchange rate between two currency"""
    url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        data = res.json()
    rates = data["rates"]
    return rates[to_currency.upper()]


@mcp.tool()
async def multiply_numbers(a: float, b: float) -> float:
    """multiply two numbers"""
    return a * b


@mcp.tool()
async def tavily_search(query: str):
    """Search the web using Tavily."""
    tavily_tool = TavilySearch(max_results=1, search_depth="basic")
    return tavily_tool.invoke(query)


@mcp.tool()
async def get_recent_emails(user_id: str, max_results: int = 5):
    """
    Get the user's most recent Gmail emails.

    Use this tool when the user asks about recent,
    latest, or newest emails.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    result = (
        service.users().messages().list(userId="me", maxResults=max_results).execute()
    )

    messages = result.get("messages", [])

    emails = []

    for message in messages:

        email = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )

        headers = email["payload"]["headers"]

        header_data = {header["name"].lower(): header["value"] for header in headers}

        emails.append(
            {
                "id": message["id"],
                "from": header_data.get("from", ""),
                "subject": header_data.get("subject", ""),
                "date": header_data.get("date", ""),
                "snippet": email.get("snippet", ""),
            }
        )

    return emails


@mcp.tool()
async def search_emails(user_id: str, query: str, max_results: int = 10):
    """
    Search Gmail using Gmail search syntax.

    Examples:
    - from:abc@gmail.com
    - subject:interview
    - is:unread
    - newer_than:7d
    - has:attachment
    - from:amazon subject:order


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    result = (
        service.users()
        .messages()
        .list(
            userId="me",
            q=query,
            maxResults=max_results,
        )
        .execute()
    )

    messages = result.get("messages", [])
    emails = []

    for message in messages:

        email = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=[
                    "From",
                    "To",
                    "Subject",
                    "Date",
                ],
            )
            .execute()
        )

        headers = {
            h["name"].lower(): h["value"] for h in email["payload"].get("headers", [])
        }

        emails.append(
            {
                "id": message["id"],
                "thread_id": email.get("threadId"),
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": email.get("snippet", ""),
                "labels": email.get("labelIds", []),
            }
        )

    return emails


@mcp.tool()
async def get_email(user_id: str, message_id: str):
    """
    Get the full content of a Gmail message using its message ID.
    Use after search_emails/get_recent_emails when the complete email
    body is required.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    message = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full",
        )
        .execute()
    )

    headers = {
        h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])
    }

    def extract_body(payload):

        body = payload.get("body", {})

        if body.get("data"):
            return base64.urlsafe_b64decode(body["data"]).decode(
                "utf-8", errors="ignore"
            )

        for part in payload.get("parts", []):

            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data")

                if data:
                    return base64.urlsafe_b64decode(data).decode(
                        "utf-8", errors="ignore"
                    )

        # Fallback for nested multipart messages
        for part in payload.get("parts", []):
            result = extract_body(part)

            if result:
                return result

        return ""

    return {
        "id": message["id"],
        "thread_id": message.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "body": extract_body(message["payload"]),
    }


@mcp.tool()
async def send_email(
    user_id: str,
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
):
    """
    Send an email to one or multiple recipients.

    attachments must contain file names returned by export tools.

    IMPORTANT:
    Use `file_name` from export_to_pdf/export_to_excel.
    NEVER use `file_url`.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    message = EmailMessage()

    message["To"] = ", ".join(to)
    message["Subject"] = subject
    message.set_content(body)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    EXPORT_DIR = os.path.join(BASE_DIR, "exports")
    USER_UPLOAD_DIR = os.path.join(BASE_DIR, "exports", user_id)

    if attachments:

        for attachment in attachments:

            # Prevent LLM from creating its own path
            file_name = os.path.basename(attachment)

            # Check the user's own uploads first, then shared export/attachment files.
            candidate_paths = [
                os.path.join(USER_UPLOAD_DIR, file_name),
                os.path.join(EXPORT_DIR, file_name),
            ]
            file_path = next((p for p in candidate_paths if os.path.exists(p)), None)

            if not file_path:
                return {
                    "success": False,
                    "error": "attachment_not_found",
                    "message": f"Attachment not found: {file_name}",
                }

            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type:
                main_type, sub_type = mime_type.split("/", 1)
            else:
                main_type = "application"
                sub_type = "octet-stream"

            with open(file_path, "rb") as file:
                file_data = file.read()

            # THIS creates the real Gmail attachment
            message.add_attachment(
                file_data,
                maintype=main_type,
                subtype=sub_type,
                filename=file_name,
            )

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    result = (
        service.users()
        .messages()
        .send(
            userId="me",
            body={"raw": encoded_message},
        )
        .execute()
    )

    return {
        "success": True,
        "message_id": result["id"],
        "thread_id": result.get("threadId"),
        "to": to,
        "subject": subject,
        "attachments": [os.path.basename(file) for file in (attachments or [])],
    }


@mcp.tool()
async def create_draft(user_id: str, to: str, subject: str, body: str):
    """
    Create a Gmail draft without sending it.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    message = EmailMessage()

    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    draft = (
        service.users()
        .drafts()
        .create(
            userId="me",
            body={"message": {"raw": encoded_message}},
        )
        .execute()
    )

    return {
        "success": True,
        "draft_id": draft["id"],
        "message_id": draft["message"]["id"],
        "to": to,
        "subject": subject,
    }


@mcp.tool()
async def mark_email_read(user_id: str, message_id: str):
    """Mark a Gmail message as read.

    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()

    return {
        "success": True,
        "message_id": message_id,
        "status": "read",
    }


@mcp.tool()
async def mark_email_unread(user_id: str, message_id: str):
    """Mark a Gmail message as unread.

    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"addLabelIds": ["UNREAD"]},
    ).execute()

    return {
        "success": True,
        "message_id": message_id,
        "status": "unread",
    }


@mcp.tool()
async def get_upcoming_events(user_id: str, max_results: int = 10):
    """
    Get the user's upcoming Google Calendar events.

    Use when the user asks about upcoming meetings,
    appointments, events, or schedule.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_calendar_service(user_id)

    now = datetime.now(timezone.utc).isoformat()

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = []

    for event in result.get("items", []):
        events.append(
            {
                "id": event["id"],
                "summary": event.get("summary", "No title"),
                "start": event.get("start", {}),
                "end": event.get("end", {}),
                "location": event.get("location"),
                "description": event.get("description"),
            }
        )

    return events


@mcp.tool()
async def create_calendar_event(
    user_id: str,
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
):
    """
    Create a Google Calendar event.

    start_time and end_time must be ISO 8601 datetime strings.

    Example:
    2026-07-24T15:00:00+05:30


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_calendar_service(user_id)

    event = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "Asia/Kolkata",
        },
    }

    created_event = (
        service.events()
        .insert(
            calendarId="primary",
            body=event,
        )
        .execute()
    )

    return {
        "success": True,
        "event_id": created_event["id"],
        "title": created_event.get("summary"),
        "start": created_event.get("start"),
        "end": created_event.get("end"),
    }


@mcp.tool()
async def update_calendar_event(
    user_id: str,
    event_id: str,
    title: str = None,
    start_time: str = None,
    end_time: str = None,
    description: str = None,
):
    """Update an existing Google Calendar event.

    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_calendar_service(user_id)

    event = (
        service.events()
        .get(
            calendarId="primary",
            eventId=event_id,
        )
        .execute()
    )

    if title is not None:
        event["summary"] = title

    if description is not None:
        event["description"] = description

    if start_time is not None:
        event["start"] = {
            "dateTime": start_time,
            "timeZone": "Asia/Kolkata",
        }

    if end_time is not None:
        event["end"] = {
            "dateTime": end_time,
            "timeZone": "Asia/Kolkata",
        }

    updated = (
        service.events()
        .update(
            calendarId="primary",
            eventId=event_id,
            body=event,
        )
        .execute()
    )

    return {
        "success": True,
        "event_id": updated["id"],
        "title": updated.get("summary"),
        "start": updated.get("start"),
        "end": updated.get("end"),
    }


@mcp.tool()
async def delete_calendar_event(user_id: str, event_id: str):
    """Delete an existing Google Calendar event.

    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_calendar_service(user_id)

    service.events().delete(
        calendarId="primary",
        eventId=event_id,
    ).execute()

    return {
        "success": True,
        "event_id": event_id,
        "message": "Calendar event deleted successfully.",
    }


@mcp.tool()
def export_to_pdf(
    content: str,
    file_name: str = "document",
):
    """
    Create a PDF from text.

    Returns:
    - file_name: use this when attaching the PDF with send_email
    - file_url: use this ONLY when the user wants to download the PDF

    IMPORTANT:
    If the PDF needs to be emailed as an attachment,
    pass the returned file_name to send_email attachments.
    NEVER pass file_url to send_email.
    """

    return create_pdf(
        content=content,
        file_name=file_name,
    )


@mcp.tool()
def export_to_excel(
    data: list[dict],
    file_name: str = "export",
):
    """
    Create an Excel file from structured data.

    `data` must be a list of dictionaries.

    Returns:
    - file_name: use this when attaching the Excel file with send_email
    - file_url: use this ONLY when the user wants to download the Excel file

    IMPORTANT:
    If the Excel file needs to be emailed as an attachment,
    pass the returned file_name to send_email attachments.
    NEVER pass file_url to send_email.

    Example data:
    [
        {"name": "Ahmed", "email": "a@gmail.com"},
        {"name": "John", "email": "j@gmail.com"}
    ]
    """

    return create_excel(
        data=data,
        file_name=file_name,
    )


@mcp.tool()
async def create_google_meet(user_id: str):
    """
    Create a new Google Meet meeting space.

    Use this tool when the user asks to:
    - create a Google Meet
    - generate a Google Meet link
    - start a new Meet space

    Returns the Google Meet URL and meeting code.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_meet_service(user_id)

    space = service.spaces().create(body={}).execute()

    return {
        "success": True,
        "space_name": space.get("name"),
        "meeting_uri": space.get("meetingUri"),
        "meeting_code": space.get("meetingCode"),
    }


@mcp.tool()
async def get_google_meet(user_id: str, space_name: str):
    """
    Get information about a Google Meet space.

    space_name should be the resource name returned by
    create_google_meet, for example: spaces/abc123


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_meet_service(user_id)

    space = service.spaces().get(name=space_name).execute()

    return {
        "success": True,
        "space_name": space.get("name"),
        "meeting_uri": space.get("meetingUri"),
        "meeting_code": space.get("meetingCode"),
        "config": space.get("config"),
        "active_conference": space.get("activeConference"),
    }


@mcp.tool()
async def end_google_meet(user_id: str, space_name: str):
    """
    End the active conference in a Google Meet space.

    Use only when the user explicitly asks to end/stop
    an active Google Meet meeting.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_meet_service(user_id)

    (
        service.spaces()
        .endActiveConference(
            name=space_name,
            body={},
        )
        .execute()
    )

    return {
        "success": True,
        "space_name": space_name,
        "message": "Google Meet conference ended.",
    }


@mcp.tool()
async def list_email_attachments(user_id: str, message_id: str):
    """
    List all attachments in a Gmail message (filename, mimeType, size, attachment_id).

    Use this before download_email_attachment to discover what's
    available and get the attachment_id needed to download it.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    message = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    attachments = []

    def walk_parts(payload):
        filename = payload.get("filename")
        body = payload.get("body", {})

        if filename and body.get("attachmentId"):
            attachments.append(
                {
                    "filename": filename,
                    "mime_type": payload.get("mimeType", "application/octet-stream"),
                    "size": body.get("size", 0),
                    "attachment_id": body["attachmentId"],
                }
            )

        for part in payload.get("parts", []):
            walk_parts(part)

    walk_parts(message["payload"])

    return {
        "message_id": message_id,
        "attachments": attachments,
    }


@mcp.tool()
async def download_email_attachment(
    user_id: str,
    message_id: str,
    attachment_id: str,
    file_name: str,
):
    """
    Download a Gmail attachment and save it locally.

    Use list_email_attachments first to get the attachment_id
    and the original filename (pass that as file_name, or your
    own name — the original extension will be kept if you don't
    include one).

    Returns:
    - file_name: use this when attaching the file with send_email
    - file_path: local path where the file was saved

    IMPORTANT:
    Use the returned `file_name` with send_email attachments,
    the same way export_to_pdf/export_to_excel results are used.


    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """

    service = await get_gmail_service(user_id)

    attachment = (
        service.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=message_id, id=attachment_id)
        .execute()
    )

    file_data = base64.urlsafe_b64decode(attachment["data"])

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Downloaded attachments go in this user's own folder so they can never
    # collide with, or be read by, another user's files.
    USER_UPLOAD_DIR = os.path.join(BASE_DIR, "exports", user_id)
    os.makedirs(USER_UPLOAD_DIR, exist_ok=True)

    # Prevent LLM from writing outside USER_UPLOAD_DIR
    safe_name = os.path.basename(file_name)
    file_path = os.path.join(USER_UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(file_data)

    return {
        "success": True,
        "file_name": safe_name,
        "file_path": file_path,
        "size": len(file_data),
    }


def _read_pdf(path, max_pages=20):
    reader = PdfReader(path)
    pages_to_read = min(len(reader.pages), max_pages)
    text = "\n\n".join(
        f"--- Page {i+1} ---\n{(reader.pages[i].extract_text() or '').strip()}"
        for i in range(pages_to_read)
    )
    return text, {"total_pages": len(reader.pages), "pages_read": pages_to_read}


def _read_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return text, {"paragraphs": len(doc.paragraphs)}


def _read_csv(path, max_rows=200):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(", ".join(row))
    text = "\n".join(rows)
    return text, {"rows_read": len(rows)}


def _read_xlsx(path, max_rows=200):
    wb = load_workbook(path, read_only=True, data_only=True)
    chunks = []
    meta = {"sheets": wb.sheetnames}
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        chunks.append(f"--- Sheet: {sheet_name} ---")
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= max_rows:
                chunks.append(f"... (truncated after {max_rows} rows)")
                break
            chunks.append(", ".join("" if c is None else str(c) for c in row))
    return "\n".join(chunks), meta


def _read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text, {}


READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".csv": _read_csv,
    ".xlsx": _read_xlsx,
    ".xls": _read_xlsx,
    ".txt": _read_txt,
}


@mcp.tool()
async def read_file(user_id: str, file_name: str):
    """
    Extract and return the text/data content of an uploaded or attached
    file so it can be read, summarized, or answered questions about.

    Supports: .pdf, .docx, .csv, .xlsx, .xls, .txt

    Image files (.png, .jpg, .jpeg) are NOT supported by this tool —
    it cannot extract text from images.

    file_name must be a file already available — e.g. one just uploaded
    by the user, or returned by export_to_pdf / export_to_excel /
    download_email_attachment.

    Use this whenever the user asks you to read, open, summarize,
    explain, or answer questions about an uploaded or attached file.

    Note: user_id is injected automatically by the system based on who is
    signed in. Never ask the user for it or attempt to supply a value.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EXPORT_DIR = os.path.join(BASE_DIR, "exports")
    USER_UPLOAD_DIR = os.path.join(BASE_DIR, "exports", user_id)

    safe_name = os.path.basename(file_name)
    ext = os.path.splitext(safe_name)[1].lower()

    # Check the user's own uploads first, then shared export/attachment files
    # (those already carry an unguessable uuid prefix, see app/tools/export.py).
    candidate_paths = [
        os.path.join(USER_UPLOAD_DIR, safe_name),
        os.path.join(EXPORT_DIR, safe_name),
    ]
    file_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not file_path:
        return {
            "success": False,
            "error": "file_not_found",
            "message": f"Could not find '{safe_name}'. "
            f"Make sure it was uploaded first.",
        }

    reader_fn = READERS.get(ext)
    if not reader_fn:
        return {
            "success": False,
            "error": "unsupported_file_type",
            "message": f"Cannot read '{ext}' files. Supported types: "
            f"{', '.join(READERS.keys())}. Image files are not readable as text.",
        }

    try:
        text, meta = reader_fn(file_path)
    except Exception as e:
        return {
            "success": False,
            "error": "read_failed",
            "message": f"Failed to read '{safe_name}': {str(e)}",
        }

    truncated = len(text) > MAX_CHARS
    return {
        "success": True,
        "file_name": safe_name,
        "file_type": ext,
        "truncated": truncated,
        "text": text[:MAX_CHARS],
        **meta,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
