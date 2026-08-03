import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
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


def is_render():
    return os.getenv("RENDER") == "true" or os.getenv("RENDER_SERVICE_ID") is not None


def restore_token():
    """Restore token.json from environment variable on Render."""
    if os.path.exists("token.json"):
        return

    token = os.getenv("GOOGLE_TOKEN_JSON")

    if token:
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(token)


def get_google_credentials():

    restore_token()

    print("=" * 60)
    print("Running on Render :", is_render())
    print("Current directory :", os.getcwd())
    print("token.json exists :", os.path.exists("token.json"))
    print("credentials exists:", os.path.exists("credentials.json"))
    print("GOOGLE_TOKEN_JSON :", bool(os.getenv("GOOGLE_TOKEN_JSON")))
    print("=" * 60)

    creds = None

    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file(
                "token.json",
                SCOPES,
            )
        except Exception as e:
            print("Failed to load token.json:", e)
            creds = None

    # Already valid
    if creds and creds.valid:
        return creds

    # Refresh expired access token
    if creds and creds.expired and creds.refresh_token:

        try:
            print("Refreshing Google access token...")

            creds.refresh(Request())

            with open("token.json", "w") as f:
                f.write(creds.to_json())

            return creds

        except RefreshError as e:

            print("Token refresh failed:", str(e))

            # Local machine → login again
            if not is_render():

                if os.path.exists("token.json"):
                    os.remove("token.json")

                print("Running local OAuth flow...")

                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    SCOPES,
                )

                creds = flow.run_local_server(port=0)

                with open("token.json", "w") as f:
                    f.write(creds.to_json())

                return creds

            # Render → cannot login
            raise RuntimeError(
                "GOOGLE_TOKEN_JSON is invalid or revoked. "
                "Generate a new token locally and update "
                "the GOOGLE_TOKEN_JSON environment variable."
            )

    # No token found
    if not is_render():

        print("Running local OAuth flow...")

        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json",
            SCOPES,
        )

        creds = flow.run_local_server(port=0)

        with open("token.json", "w") as f:
            f.write(creds.to_json())

        return creds

    raise RuntimeError("No valid GOOGLE_TOKEN_JSON found on Render.")


def get_gmail_service():
    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)
