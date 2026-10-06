"""PostgreSQL job and result tables; source files live in storage."""

from enum import Enum
from datetime import datetime
from typing import Optional, Any
from sqlmodel import SQLModel, Field
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

from ..util import now_toronto


class JobStatus(str, Enum):
    """Document processing states."""
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class PageStatus(str, Enum):
    """Legacy page states; the current worker does not create JobPage rows."""
    pending = "pending"
    done = "done"


# One job per uploaded document.
class Job(SQLModel, table=True):
    """Store upload metadata, file location and processing status."""
    id: Optional[int] = Field(default=None, primary_key=True)
    status: JobStatus = Field(default=JobStatus.uploaded, index=True)
    original_filename: str
    file_type: str
    file_path: Optional[str] = None
    page_count: Optional[int] = None
    created_at: datetime = Field(default_factory=now_toronto)
    updated_at: datetime = Field(default_factory=now_toronto)
    error_message: Optional[str] = None


class JobPage(SQLModel, table=True):
    """Legacy page table; the worker stores pages in JobResult JSON."""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    page_index: int
    width: Optional[float] = None
    height: Optional[float] = None
    status: PageStatus = Field(default=PageStatus.pending)


# Keep original OCR results when saving user edits.
class JobResult(SQLModel, table=True):
    """Keep raw output, original OCR results and user edits in separate JSONB fields."""
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id")
    page_id: Optional[int] = Field(default=None, foreign_key="jobpage.id")
    raw_json: dict[str, Any] = Field(sa_column=Column(JSONB), default_factory=dict)
    result_json: dict[str, Any] = Field(sa_column=Column(JSONB), default_factory=dict)
    edited_json: Optional[dict[str, Any]] = Field(sa_column=Column(JSONB), default=None)
    created_at: datetime = Field(default_factory=now_toronto)
    updated_at: datetime = Field(default_factory=now_toronto)
