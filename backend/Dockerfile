# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.docker.txt

FROM base AS api
COPY main.py ./
COPY src ./src
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
# Native Linux CPU wheels, including Apple Silicon's Linux/ARM64 runtime.
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 120 --retries 5 paddlepaddle==3.3.0 \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
COPY requirements.ocr.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.ocr.txt -c requirements.docker.txt \
    && pip check
COPY main.py ./
COPY src ./src
COPY tests ./tests
CMD ["celery", "-A", "src.core.celery_app:celery_app", "worker", "--pool=prefork", "--concurrency=1", "--hostname=ocr@%h", "--loglevel=info"]
