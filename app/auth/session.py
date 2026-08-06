"""
Minimal JWT session handling. We issue a JWT after a successful Google
login and store it in an httpOnly cookie. No separate password system —
Google login IS the account system.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Request, HTTPException

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set in .env")

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30
SESSION_COOKIE_NAME = "session_token"


def create_session_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> str:
    """Returns user_id, or raises HTTPException(401) if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail="Session expired. Please sign in again."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session.")


def get_current_user_id(request: Request) -> str:
    """FastAPI dependency: reads the session cookie and returns the user_id."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return decode_session_token(token)


def get_current_user_id_optional(request: Request) -> str | None:
    """
    Same as get_current_user_id, but returns None for guests instead of
    raising. Used on endpoints that should work for signed-out visitors
    (e.g. /chat/stream, where a guest can chat normally and only gets
    blocked at the point a Google tool is actually needed).
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        return decode_session_token(token)
    except HTTPException:
        # expired/invalid cookie — treat as a guest rather than erroring
        return None
