import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/meetings.space.readonly",
]


def _restore_token_from_env():
    """
    On Render (and any ephemeral filesystem), token.json won't exist
    after a deploy, and there's no browser to run the OAuth consent
    flow. Restore it from an env var set once from your local token.json.
    """
    if os.path.exists("token.json"):
        return

    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if token_json:
        with open("token.json", "w") as f:
            f.write(token_json)


def get_google_credentials():

    _restore_token_from_env()

    credentials = None

    if os.path.exists("token.json"):
        credentials = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES,
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )
            credentials = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(credentials.to_json())

    return credentials


def get_gmail_service():
    credentials = get_google_credentials()
    return build("gmail", "v1", credentials=credentials)
