"""Provide request-scoped database sessions."""

from .engine import engine
from sqlmodel import Session


def get_session() -> Session:
    """Yield a session and close it after the request; callers commit their own changes."""
    with Session(engine) as session:
        yield session
