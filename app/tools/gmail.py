import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = [
    # gmail
    "https://www.googleapis.com/auth/gmail.readonly",  # read emails
    "https://www.googleapis.com/auth/gmail.compose",  # create/update drafts
    "https://www.googleapis.com/auth/gmail.send",  # send emails (incl. replies)
    "https://www.googleapis.com/auth/gmail.modify",  # mark read/unread, labels, trash, etc.
    # Calendar
    "https://www.googleapis.com/auth/calendar.events",  # read, create, modify, and delete events.
]


def get_google_credentials():

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

    return build(
        "gmail",
        "v1",
        credentials=credentials,
    )
