"""
FairLens Backend — FastAPI Application Entry Point
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import upload, analyze, mitigate, report, gemini_chat

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise in-memory session store on startup
    app.state.sessions = {}
    yield
    # Nothing to clean up on shutdown for in-memory store


def create_app() -> FastAPI:
    app = FastAPI(
        title="FairLens API",
        description=(
            "Detect, explain, and mitigate bias in datasets and ML models "
            "using Google Gemini 2.0 Flash."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(upload.router, prefix="/api")
    app.include_router(analyze.router, prefix="/api")
    app.include_router(mitigate.router, prefix="/api")
    app.include_router(report.router, prefix="/api")
    app.include_router(gemini_chat.router, prefix="/api")

    # ── Health check ─────────────────────────────────────────────────────────
    @app.get("/api/health", tags=["Health"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
