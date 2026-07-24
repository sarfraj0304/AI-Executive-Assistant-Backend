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

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

mcp = FastMCP("chatbot_mcp")


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
def get_recent_emails(max_results: int = 5):
    """
    Get the user's most recent Gmail emails.

    Use this tool when the user asks about recent,
    latest, or newest emails.
    """

    service = get_gmail_service()

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
def search_emails(query: str, max_results: int = 10):
    """
    Search Gmail using Gmail search syntax.

    Examples:
    - from:abc@gmail.com
    - subject:interview
    - is:unread
    - newer_than:7d
    - has:attachment
    - from:amazon subject:order
    """

    service = get_gmail_service()

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
def get_email(message_id: str):
    """
    Get the full content of a Gmail message using its message ID.
    Use after search_emails/get_recent_emails when the complete email
    body is required.
    """

    service = get_gmail_service()

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
def send_email(to: str, subject: str, body: str):
    """
    Send an email.

    Use only when the user explicitly asks to send an email.
    """

    service = get_gmail_service()

    message = EmailMessage()

    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

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
    }


@mcp.tool()
def create_draft(to: str, subject: str, body: str):
    """
    Create a Gmail draft without sending it.
    """

    service = get_gmail_service()

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
def mark_email_read(message_id: str):
    """Mark a Gmail message as read."""

    service = get_gmail_service()

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
def mark_email_unread(message_id: str):
    """Mark a Gmail message as unread."""

    service = get_gmail_service()

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
def get_upcoming_events(max_results: int = 10):
    """
    Get the user's upcoming Google Calendar events.

    Use when the user asks about upcoming meetings,
    appointments, events, or schedule.
    """

    service = get_calendar_service()

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
def create_calendar_event(
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
    """

    service = get_calendar_service()

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
def update_calendar_event(
    event_id: str,
    title: str = None,
    start_time: str = None,
    end_time: str = None,
    description: str = None,
):
    """Update an existing Google Calendar event."""

    service = get_calendar_service()

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
def delete_calendar_event(event_id: str):
    """Delete an existing Google Calendar event."""

    service = get_calendar_service()

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
    Convert text/content into a downloadable PDF file.

    Use when the user asks to export, save,
    convert or download content as PDF.
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
    Convert structured data into a downloadable Excel file.

    data must be a list of dictionaries.

    Example:
    [
        {"name": "Ahmed", "email": "a@gmail.com"},
        {"name": "John", "email": "j@gmail.com"}
    ]
    """

    return create_excel(
        data=data,
        file_name=file_name,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
