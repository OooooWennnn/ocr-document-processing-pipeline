"""Record child-process failures from the surviving Celery parent."""

import logging
from billiard.einfo import ExceptionWithTraceback
from billiard.exceptions import WorkerLostError, TimeLimitExceeded, Terminated
from celery import Task
from celery.worker.request import Request
from sqlmodel import Session
from src.db.engine import engine
from src.models.job_model import Job, JobStatus
from src.util import now_toronto

logger = logging.getLogger(__name__)


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
