from googleapiclient.discovery import build
from app.tools.gmail import get_google_credentials


def get_meet_service():
    credentials = get_google_credentials()

    return build(
        "meet",
        "v2",
        credentials=credentials,
    )
