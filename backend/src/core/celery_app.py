"""Redis queue settings and OCR child-process limits."""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required. Start the API or worker with Docker Compose.")

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL, include=["src.tasks.ocr_tasks"])
celery_app.conf.broker_connection_retry_on_startup = True

# 한 번에 한 작업만 예약하고, 모델이 잡고 있는 메모리를 주기적으로 해제한다.
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.worker_max_tasks_per_child = int(os.getenv("OCR_WORKER_MAX_TASKS_PER_CHILD", "5"))

if celery_app.conf.worker_max_tasks_per_child < 1:
    raise ValueError("OCR_WORKER_MAX_TASKS_PER_CHILD must be at least 1.")
