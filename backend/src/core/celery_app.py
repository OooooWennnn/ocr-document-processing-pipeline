"""Redis queue settings and OCR child-process limits."""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required. Start the API or worker with Docker Compose.")

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL, include=["src.tasks.ocr_tasks"])
celery_app.conf.broker_connection_retry_on_startup = True

# Limit reservations and recycle children after five tasks, including warmup.
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.worker_max_tasks_per_child = 5
