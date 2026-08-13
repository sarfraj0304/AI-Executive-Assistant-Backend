import code
import os
import json

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
import google.oauth2.id_token
import google.auth.transport.requests

from app.db import upsert_user_from_google
from app.auth.session import create_session_token, SESSION_COOKIE_NAME, JWT_EXPIRE_DAYS

router = APIRouter(prefix="/auth/google", tags=["auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Every scope your tools need, requested up-front at signup.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/meetings.space.created",
    "https://www.googleapis.com/auth/meetings.space.readonly",
]


def _build_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [OAUTH_REDIRECT_URI],
        }
    }
    return Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=OAUTH_REDIRECT_URI
    )


@router.get("/login")
async def google_login(request: Request):
    flow = _build_flow()
    print("Redirect URI:", OAUTH_REDIRECT_URI)
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    request.session["oauth_state"] = state
    request.session["code_verifier"] = flow.code_verifier

    return RedirectResponse(auth_url)


@router.get("/callback")
async def google_callback(request: Request):
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        return RedirectResponse(f"{FRONTEND_URL}/login?error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    flow = _build_flow()
    verifier = request.session.get("code_verifier")

    if verifier is None:
        raise HTTPException(
            status_code=400,
            detail="Missing PKCE verifier. OAuth session expired.",
        )

    flow.code_verifier = verifier
    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Verify + decode the ID token to get the user's stable Google profile.
    request_adapter = google.auth.transport.requests.Request()
    id_info = google.oauth2.id_token.verify_oauth2_token(
        credentials.id_token,
        request_adapter,
        GOOGLE_CLIENT_ID,
        clock_skew_in_seconds=5,
    )

    google_profile = {
        "sub": id_info["sub"],
        "email": id_info.get("email"),
        "name": id_info.get("name"),
        "picture": id_info.get("picture"),
    }

    token_dict = json.loads(credentials.to_json())

    user = await upsert_user_from_google(google_profile, token_dict)

    session_token = create_session_token(user["_id"])

    response = RedirectResponse(f"{FRONTEND_URL}/")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=JWT_EXPIRE_DAYS * 24 * 60 * 60,
    )
    return response


@router.post("/logout")
async def google_logout():
    response = RedirectResponse(f"{FRONTEND_URL}/")
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=True,
        samesite="none",
    )
    return response
