from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.graph import init_graph
from app.routes import router
from app.auth.google_oauth import router as google_auth_router
from fastapi.staticfiles import StaticFiles
import os
from starlette.middleware.sessions import SessionMiddleware
import secrets


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_graph()  # builds graph + starts MCP subprocess once
    yield


app = FastAPI(title="AI Executive Assistant", lifespan=lifespan)
os.makedirs("exports", exist_ok=True)
app.mount(
    "/exports",
    StaticFiles(directory="exports"),
    name="exports",
)
# Session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", secrets.token_hex(32)),
)
# allow_credentials=True + explicit origins (not "*") is required for the
# session cookie to be sent/received cross-origin between frontend and API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "https://aiexecutiveassistant.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(google_auth_router)
