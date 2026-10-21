"""FastAPI setup, database initialization and health checks."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel

from src.database import engine
import src.models

from src.api import api_router

app = FastAPI()

configured_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(api_router, prefix="/api/v1")

SQLModel.metadata.create_all(engine)


@app.get("/health", include_in_schema=False)
def health():
    """Check HTTP availability; this does not check OCR model readiness."""
    return {"status": "ok"}
