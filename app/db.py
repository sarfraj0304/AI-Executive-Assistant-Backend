import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from bson import ObjectId
from cryptography.fernet import Fernet
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("MONGODB_DB_NAME")

# Fernet key used to encrypt Google refresh/access tokens at rest.
# Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
TOKEN_ENCRYPTION_KEY = os.getenv("TOKEN_ENCRYPTION_KEY")

if not TOKEN_ENCRYPTION_KEY:
    raise RuntimeError(
        "TOKEN_ENCRYPTION_KEY is not set. Generate one with:\n"
        '  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n'
        "and add it to your .env file."
    )

_fernet = Fernet(TOKEN_ENCRYPTION_KEY.encode())

_client = AsyncIOMotorClient(MONGO_URL)
_db = _client[DB_NAME]
users_collection = _db["users"]


def encrypt_tokens(token_dict: dict) -> str:
    raw = json.dumps(token_dict).encode()
    return _fernet.encrypt(raw).decode()


def decrypt_tokens(blob: str) -> dict:
    raw = _fernet.decrypt(blob.encode())
    return json.loads(raw.decode())


async def upsert_user_from_google(google_profile: dict, token_dict: dict) -> dict:
    """
    Create or update a user record after a successful Google OAuth flow.
    `token_dict` is Credentials.to_json() parsed back into a dict.
    Returns the stored user document (with _id as str).
    """
    now = datetime.now(timezone.utc)
    encrypted = encrypt_tokens(token_dict)

    result = await users_collection.find_one_and_update(
        {"google_id": google_profile["sub"]},
        {
            "$set": {
                "email": google_profile.get("email"),
                "name": google_profile.get("name"),
                "picture": google_profile.get("picture"),
                "google_tokens": encrypted,
                "updated_at": now,
            },
            "$setOnInsert": {
                "google_id": google_profile["sub"],
                "created_at": now,
            },
        },
        upsert=True,
        return_document=True,
    )
    result["_id"] = str(result["_id"])
    return result


async def get_user_by_id(user_id: str) -> dict | None:
    try:
        doc = await users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def save_refreshed_tokens(user_id: str, token_dict: dict) -> None:
    """Call this after google Credentials auto-refreshes, to persist the new access token."""
    encrypted = encrypt_tokens(token_dict)
    await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {
            "$set": {
                "google_tokens": encrypted,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
