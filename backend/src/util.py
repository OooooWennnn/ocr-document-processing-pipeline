"""Timestamps for job records."""

from datetime import datetime
from zoneinfo import ZoneInfo


def now_toronto() -> datetime:
    """Return the current time in the Toronto timezone."""
    return datetime.now(ZoneInfo("America/Toronto"))
