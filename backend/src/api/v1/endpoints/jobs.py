"""Document uploads, job status, source previews and saved edits."""

import os
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi.responses import FileResponse
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field, ConfigDict
from sqlmodel import Session, select

from src.db.session import get_session
from src.models.job_model import Job, JobResult, JobStatus
from src.services.result_editing import apply_document_edits, present_document
from src.tasks.ocr_tasks import process_job
from src.util import now_toronto
from src.services.document_input import validate_pdf, preview_path, resolve_source_path

router = APIRouter()
MAX_BYTES = 50 * 1024 * 1024

MAX_PIXELS = 25_000_000
STORAGE_ROOT = Path(os.getenv("OCR_STORAGE_DIR", str(Path(__file__).resolve().parents[5] / "storage"))).expanduser().resolve()


class CellEdit(BaseModel):
    """A row, column and replacement value for legacy table edits."""
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    value: str = Field(max_length=10000)


class TextEdit(BaseModel):
    """A region ID and replacement text."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=100)
    value: str = Field(max_length=10000)


class PageRegionEdit(BaseModel):
    """Edits for one page, using a zero-based page index."""
    model_config = ConfigDict(extra="forbid")
    page_index: int = Field(ge=0)
    regions: list[TextEdit] = Field(min_length=1, max_length=10000)


class ResultEdit(BaseModel):
    """Validate edit payloads; pages is current, cells/texts support older clients."""
    model_config = ConfigDict(extra="forbid")
    page_index: int = Field(default=0, ge=0)
    table_id: str | None = None
    cells: list[CellEdit] = Field(default_factory=list, max_length=10000)
    texts: list[TextEdit] = Field(default_factory=list, max_length=10000)
    regions: list[TextEdit] = Field(default_factory=list, max_length=10000)
    pages: list[PageRegionEdit] = Field(default_factory=list, max_length=30)


def get_job_or_404(session, job_id):
    """Find a job by its primary key or raise HTTP 404."""
    job = session.get(Job, job_id)

    if job is None:
        raise HTTPException(404, "Job not found.")

    return job


def get_result_or_404(session, job_id):
    """Find the latest result by job_id or raise HTTP 404."""
    result = session.exec(select(JobResult).where(JobResult.job_id == job_id).order_by(JobResult.id.desc())).first()

    if result is None:
        raise HTTPException(404, "Saved result not found.")

    return result


@router.get("/{job_id}")
def get_job(job_id: int, session: Session = Depends(get_session)):
    """Return job metadata, status and any error message."""
    return get_job_or_404(session, job_id)


@router.get("/{job_id}/image")
def get_image(job_id: int, page_index: int = 0, session: Session = Depends(get_session)):
    """Return the source image or a zero-based PDF page preview; missing previews raise 404."""
    job = get_job_or_404(session, job_id)

    if not 0 <= page_index < (job.page_count or 1):
        raise HTTPException(404, "Page not found.")

    source = resolve_source_path(job.file_path) if job.file_path else Path("")
    path = source

    if job.file_path and job.file_type == "application/pdf":
        path = preview_path(source, page_index)

    if not path.is_file():
        raise HTTPException(404, "Page preview is not available yet.")

    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@router.get("/{job_id}/result")
def get_result(job_id: int, session: Session = Depends(get_session)):
    """Return saved edits before OCR originals, with job status and page progress."""
    job = get_job_or_404(session, job_id)
    data = session.exec(select(JobResult).where(JobResult.job_id == job_id).order_by(JobResult.id.desc())).first()
    payload = {}

    if data and data.result_json:
        saved_result = data.edited_json if data.edited_json is not None else data.result_json
        payload = present_document(saved_result)

    return {**payload, "status": job.status, "message": job.error_message, "page_count": job.page_count or 1}


@router.post("/", status_code=201)
async def upload_job(file: UploadFile, session: Session = Depends(get_session)):
    """Validate and save a document, create its job and queue the ID. OCR runs separately."""
    content = await file.read(MAX_BYTES + 1)

    await file.close()

    if not content or len(content) > MAX_BYTES:
        raise HTTPException(413, "Choose a non-empty document of up to 50 MB.")

    is_pdf = content.startswith(b"%PDF-")
    page_count = 1
    extension = ".pdf" if is_pdf else ".png"
    destination = STORAGE_ROOT / "uploads" / f"{uuid4().hex}{extension}"
    image = None

    if is_pdf:
        try:
            page_count = validate_pdf(content)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    else:
        if not (content.startswith(b"\x89PNG\r\n\x1a\n") or content.startswith(b"\xff\xd8\xff")):
            raise HTTPException(415, "Only PNG, JPEG and PDF documents are supported.")

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "Images must be 10 MB or smaller.")

        try:
            with Image.open(BytesIO(content)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise HTTPException(413, "Reduce the image to 25 million pixels or fewer.")

                source.verify()

            with Image.open(BytesIO(content)) as source:
                upright = ImageOps.exif_transpose(source).convert("RGBA")
                normalized = Image.alpha_composite(Image.new("RGBA", upright.size, "white"), upright).convert("RGB")
                image = cv2.cvtColor(np.asarray(normalized), cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
            raise HTTPException(422, "The image could not be read.")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        if is_pdf:
            destination.write_bytes(content)
        elif not cv2.imwrite(str(destination), image):
            raise OSError("image write failed")

        job_data = {
            "original_filename": file.filename or "image",
            "file_type": "application/pdf" if is_pdf else "image/png",
            "file_path": str(destination),
            "page_count": page_count,
            "status": JobStatus.queued,
        }
        job = Job(**job_data)

        session.add(job)
        session.commit()
        session.refresh(job)
    except Exception:
        session.rollback()
        destination.unlink(missing_ok=True)
        raise HTTPException(500, "The document could not be saved.")

    try:
        process_job.delay(job.id)
    except Exception:
        job.status = JobStatus.failed
        job.error_message = "The processing queue is unavailable. Please try uploading again shortly."
        job.updated_at = now_toronto()

        session.add(job)
        session.commit()
        raise HTTPException(503, job.error_message)

    return {"job_id": job.id, "status": job.status}


@router.patch("/{job_id}/result")
def update_result(job_id: int, updated_result: ResultEdit, session: Session = Depends(get_session)):
    """Validate edits and save edited_json without replacing the original OCR result."""
    job = get_job_or_404(session, job_id)

    if job.status != JobStatus.done:
        raise HTTPException(409, "Only completed jobs can be edited.")

    result = get_result_or_404(session, job_id)

    if not (updated_result.cells or updated_result.texts or updated_result.regions or updated_result.pages):
        raise HTTPException(422, "Supply at least one text or cell edit.")

    try:
        saved_result = result.edited_json if result.edited_json is not None else result.result_json
        result.edited_json = apply_document_edits(saved_result, updated_result)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    result.updated_at = now_toronto()

    session.add(result)
    session.commit()

    return {"status": "done", **present_document(result.edited_json)}
