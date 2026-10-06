"""Process queued documents page by page and persist progress and results."""

import logging
import os
import time
from sqlmodel import Session, select
import cv2
import numpy as np
from celery.signals import worker_ready, worker_process_init
from src.core.celery_app import celery_app as app
from src.db.engine import engine
from src.models.job_model import Job, JobResult, JobStatus
from src.util import now_toronto
from src.tasks.ocr_runtime import get_extractor
from src.tasks.ocr_task_base import DocumentTask
from src.services.document_input import iter_pages, preview_path, resolve_source_path
from src.services.table_extraction import BBox, coverage

logger = logging.getLogger(__name__)


@app.task
def warmup_model():
    """Initialize OCR in a worker child before the first document."""
    get_extractor()

    return {"ready": True}


@worker_ready.connect
def warm_model(sender=None, **kwargs):
    """Queue child warmup when enabled; initialize directly for solo workers."""
    if os.getenv("OCR_WARMUP", "1") != "1" or sender is None:
        return

    if sender.pool.__class__.__module__ == "celery.concurrency.solo":
        try:
            get_extractor()
        except Exception:
            logger.exception("OCR warmup failed; jobs will retry model initialization")
    else:
        # Load Paddle inside the OCR child, after fork.
        warmup_model.delay()


@worker_process_init.connect
def reset_child_connections(**kwargs):
    """Replace inherited connection pools without closing the parent's connections."""
    engine.dispose(close=False)


def merge_native_text(page, native):
    """Update page text in place, preferring native PDF tokens and keeping OCR-only regions."""

    if not native:
        return

    boxes = [BBox(**t["bbox"]) for t in native]
    extra = []

    for text in page["texts"]:
        text_box = BBox(**text["bbox"])

        if not any(coverage(text_box, box) >= 0.5 for box in boxes):
            extra.append(text)
    texts = sorted([*native, *extra], key=lambda t: (t["bbox"]["y_min"], t["bbox"]["x_min"]))
    page["texts"] = [{**t, "id": str(i)} for i, t in enumerate(texts)]
    page["text_source"] = "pdf_text+ocr" if extra else "pdf_text"


TASK_TIME_LIMIT = max(60, int(os.getenv("OCR_TASK_TIME_LIMIT", "5400")))


@app.task(base=DocumentTask, acks_late=True, reject_on_worker_lost=False, soft_time_limit=TASK_TIME_LIMIT, time_limit=TASK_TIME_LIMIT + 60)
def process_job(job_id: int):
    """Extract pages and previews, committing progress after each page and done at the end.
    Handle Python errors here; DocumentRequest handles child termination."""
    started = time.perf_counter()

    with Session(engine) as session:
        job = session.get(Job, job_id)

        if not job or job.status == JobStatus.done:
            return

        result = session.exec(select(JobResult).where(JobResult.job_id == job_id).order_by(JobResult.id.desc())).first()

        try:
            job.status = JobStatus.processing
            job.error_message = None
            job.updated_at = now_toronto()

            session.commit()

            if result is None:
                result = JobResult(job_id=job_id)

                session.add(result)

            document = {"schema_version": 3, "page_count": job.page_count or 1, "pages": []}
            raw_pages = []
            result.result_json = document

            session.commit()

            extractor = get_extractor()

            for index, pil_image, native, native_only in iter_pages(job.file_path, job.file_type):
                image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

                if job.file_type == "application/pdf":
                    destination = preview_path(resolve_source_path(job.file_path), index)

                    destination.parent.mkdir(parents=True, exist_ok=True)

                    if not cv2.imwrite(str(destination), image):
                        raise OSError("Could not save page preview")

                if native_only:
                    raw, page = extractor.detect_document(image, native_texts=native)
                else:
                    raw, page = extractor.detect_document(image)

                merge_native_text(page, native)

                page["page_index"] = index

                document["pages"].append(page)
                raw_pages.append(raw)

                # Assign fresh JSON objects so the ORM tracks each completed page.
                result.result_json = {**document, "pages": list(document["pages"])}
                result.raw_json = {"pages": list(raw_pages)}
                result.edited_json = None
                job.updated_at = now_toronto()

                session.commit()
                logger.info("Job %s page %s/%s complete in %.1fs", job_id, index + 1, document["page_count"], page["elapsed_seconds"])

            document["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            result.result_json = document
            job.status = JobStatus.done
            job.updated_at = now_toronto()

            session.commit()
        except Exception as exc:
            logger.exception("Document job %s failed", job_id)
            session.rollback()

            job.status = JobStatus.failed
            job.updated_at = now_toronto()
            job.error_message = str(exc) if isinstance(exc, ValueError) else "Document processing failed. Check the worker logs."

            session.commit()
