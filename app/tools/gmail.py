import json

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build

from app.db import get_user_by_id, decrypt_tokens, save_refreshed_tokens

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/meetings.space.readonly",
]


async def get_google_credentials(user_id: str) -> Credentials:
    """
    Loads this user's Google credentials from MongoDB (decrypting the
    stored token blob), refreshing + persisting a new access token if
    it has expired.
    """
    user = await get_user_by_id(user_id)
    if not user or not user.get("google_tokens"):
        raise RuntimeError(
            "This account isn't connected to Google. Please sign in with "
            "Google to grant Gmail/Calendar/Meet access."
        )

    token_dict = decrypt_tokens(user["google_tokens"])
    creds = Credentials.from_authorized_user_info(token_dict, SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            raise RuntimeError(
                "Google access has expired and could not be refreshed. "
                "Please sign in with Google again to reconnect."
            ) from e

        # Persist the refreshed access token so we don't refresh every call.
        await save_refreshed_tokens(user_id, json.loads(creds.to_json()))
        return creds

    raise RuntimeError(
        "Google credentials are invalid. Please sign in with Google again."
    )


async def get_gmail_service(user_id: str):
    creds = await get_google_credentials(user_id)
    return build("gmail", "v1", credentials=creds)
