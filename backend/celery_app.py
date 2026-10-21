"""Celery 설정·작업 등록·모델 준비와 자식 프로세스 종료 처리를 관리한다."""

import logging
import os

from billiard.einfo import ExceptionWithTraceback
from billiard.exceptions import WorkerLostError, TimeLimitExceeded, Terminated
from celery import Celery, Task
from celery.signals import worker_process_init, worker_ready
from celery.worker.request import Request
from sqlmodel import Session

from src.database import engine
from src.models import Job, JobStatus
from src.services.document_job import get_extractor, process_job as process_document_job
from src.util import now_toronto

REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required. Start the API or worker with Docker Compose.")

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL, include=["celery_app"])
celery_app.conf.broker_connection_retry_on_startup = True

# 한 번에 한 작업만 예약하고, 모델이 잡고 있는 메모리를 주기적으로 해제한다.
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.worker_max_tasks_per_child = int(os.getenv("OCR_WORKER_MAX_TASKS_PER_CHILD", "5"))

if celery_app.conf.worker_max_tasks_per_child < 1:
    raise ValueError("OCR_WORKER_MAX_TASKS_PER_CHILD must be at least 1.")

logger = logging.getLogger(__name__)
TASK_TIME_LIMIT = max(60, int(os.getenv("OCR_TASK_TIME_LIMIT", "5400")))


def mark_interrupted_job(job_id, reason):
    """Mark queued or processing jobs failed; preserve completed jobs and log persistence errors."""
    try:
        with Session(engine) as session:
            job = session.get(Job, job_id)

            if job is not None and job.status in (JobStatus.processing, JobStatus.queued):
                job.status = JobStatus.failed
                job.error_message = reason
                job.updated_at = now_toronto()

                session.commit()
    except Exception:
        logger.exception("Could not persist interrupted OCR job %s", job_id)


class DocumentRequest(Request):
    """Let the parent record failures when an OCR child stops."""
    def _mark_interrupted(self, reason):
        """Read job_id from task arguments and record the interruption."""
        job_id = self.args[0] if self.args else self.kwargs.get("job_id")

        if job_id is not None:
            mark_interrupted_job(job_id, reason)

    def on_failure(self, exc_info, send_failed_event=True, return_ok=False):
        """Record child loss or termination, then run the default Celery failure handler."""
        exc = exc_info.exception

        if isinstance(exc, ExceptionWithTraceback):
            exc = exc.exc

        if isinstance(exc, (WorkerLostError, TimeLimitExceeded, Terminated)):
            self._mark_interrupted("OCR process stopped unexpectedly or exceeded its limits. Please retry the document.")

        super().on_failure(exc_info, send_failed_event=send_failed_event, return_ok=return_ok)

    def on_timeout(self, soft, timeout):
        """Record hard timeouts; leave soft-timeout handling to the child."""
        if not soft:
            self._mark_interrupted("OCR processing exceeded its time limit. Please retry a smaller document.")

        super().on_timeout(soft, timeout)


class DocumentTask(Task):
    """Use DocumentRequest for document tasks."""
    abstract = True
    Request = DocumentRequest


@celery_app.task(name="src.tasks.ocr_tasks.warmup_model")
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


@celery_app.task(name="src.tasks.ocr_tasks.process_job", base=DocumentTask, acks_late=True, reject_on_worker_lost=False, soft_time_limit=TASK_TIME_LIMIT, time_limit=TASK_TIME_LIMIT + 60)
def process_job(job_id: int):
    """Celery가 받은 작업 ID를 문서 처리 함수에 전달한다. 자식 종료는 DocumentRequest가 처리한다."""
    process_document_job(job_id)
