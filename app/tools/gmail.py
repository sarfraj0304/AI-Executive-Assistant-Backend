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


def get_file_path(filename: str) -> str | None:
    """
    Prefer local file.
    If not present, use Render Secret File.
    """
    if os.path.exists(filename):
        return filename

    secret_path = f"/etc/secrets/{filename}"

    if os.path.exists(secret_path):
        return secret_path

    return None


def get_google_credentials():

    token_path = get_file_path("token.json")
    credentials_path = get_file_path("credentials.json")

    print("=" * 60)
    print("Current directory :", os.getcwd())
    print("Token path        :", token_path)
    print("Credentials path  :", credentials_path)
    print("=" * 60)

    creds = None

    if token_path:
        try:
            creds = Credentials.from_authorized_user_file(
                token_path,
                SCOPES,
            )
        except Exception as e:
            print("Failed to load token:", e)

    # Already authenticated
    if creds and creds.valid:
        return creds

    # Refresh token
    if creds and creds.expired and creds.refresh_token:

        try:
            print("Refreshing Google token...")

            creds.refresh(Request())

            # Save only if local file exists
            if token_path == "token.json":
                with open("token.json", "w") as f:
                    f.write(creds.to_json())

            return creds

        except RefreshError as e:
            print("Refresh failed:", e)

    # Local machine -> Browser login
    if credentials_path == "credentials.json":

        print("Running OAuth login...")

        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path,
            SCOPES,
        )

        creds = flow.run_local_server(port=0)

        with open("token.json", "w") as f:
            f.write(creds.to_json())

        return creds

    # Render -> Cannot login
    raise RuntimeError(
        "Google authentication failed.\n"
        "Make sure BOTH Secret Files exist:\n"
        " - credentials.json\n"
        " - token.json"
    )


def get_gmail_service():
    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)
