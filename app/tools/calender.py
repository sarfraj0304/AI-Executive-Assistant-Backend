from googleapiclient.discovery import build
from app.tools.gmail import get_google_credentials


def get_calendar_service():

    credentials = get_google_credentials()

    return build(
        "calendar",
        "v3",
        credentials=credentials,
    )
