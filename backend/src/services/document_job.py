"""문서 읽기부터 OCR·페이지별 결과 저장까지 실행한다. 작업 큐 등록은 호출자가 담당한다."""

import logging
import time
from threading import Lock

import cv2
import numpy as np
from sqlmodel import Session, select

from src.database import engine
from src.models import Job, JobResult, JobStatus
from src.services.document_input import iter_pages, preview_path
from src.services.document_layout import covered_fraction, normalize
from src.services.document_storage import local_source, save_preview
from src.services.table_extraction import TableExtraction
from src.util import now_toronto

logger = logging.getLogger(__name__)

_extractor = None
_lock = Lock()


def get_extractor():
    """현재 프로세스의 OCR 모델을 반환한다. 잠금으로 중복 생성을 막고 생성한 모델을 재사용한다."""
    global _extractor

    with _lock:
        if _extractor is None:
            started = time.perf_counter()

            logger.info("Loading document OCR model")

            _extractor = TableExtraction()

            logger.info("Document OCR model ready in %.1fs", time.perf_counter() - started)

        return _extractor


def merge_native_text(page, native):
    """PDF 원래 텍스트와 중복되는 OCR을 제거하고 page의 텍스트 목록을 직접 갱신한다."""

    if not native:
        return

    native_boxes = [token["bbox"] for token in native]
    additional_texts = []

    for text in page["texts"]:
        contained_tokens = [token for token in native if covered_fraction(token["bbox"], [text["bbox"]]) >= 0.5]
        native_value = normalize("".join(token["value"] for token in contained_tokens))
        ocr_value = normalize(text["value"])

        # OCR 박스에 여백이 많으면 반대 방향의 포함 관계와 실제 글자도 확인한다.
        same_text = bool(ocr_value) and ocr_value in native_value

        if covered_fraction(text["bbox"], native_boxes) < 0.5 and not same_text:
            additional_texts.append(text)

    texts = sorted([*native, *additional_texts], key=lambda text: (text["bbox"]["y_min"], text["bbox"]["x_min"]))
    page["texts"] = [{**text, "id": str(index)} for index, text in enumerate(texts)]
    page["text_source"] = "pdf_text+ocr" if additional_texts else "pdf_text"


def process_job(job_id: int):
    """작업 ID로 문서를 읽고 페이지별 결과를 DB에 저장한다. 성공·실패 상태를 기록하며 별도 결과를 반환하지 않는다."""
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

            document = {
                "schema_version": 3,
                "page_count": job.page_count or 1,
                "pages": [],
            }
            raw_pages = []
            result.result_json = document

            session.commit()

            extractor = get_extractor()

            with local_source(job.file_path) as source:
                for page_index, pil_image, native_texts, native_only in iter_pages(source, job.file_type):
                    image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

                    if job.file_type == "application/pdf":
                        destination = preview_path(source, page_index)

                        destination.parent.mkdir(parents=True, exist_ok=True)

                        if not cv2.imwrite(str(destination), image):
                            raise OSError("Could not save page preview")

                        save_preview(job.file_path, destination, page_index)

                    if native_only:
                        raw, page = extractor.detect_document(image, native_texts=native_texts)
                    else:
                        raw, page = extractor.detect_document(image)

                    merge_native_text(page, native_texts)

                    page["page_index"] = page_index

                    document["pages"].append(page)
                    raw_pages.append(raw)

                    # Assign fresh JSON objects so the ORM tracks each completed page.
                    result.result_json = {**document, "pages": list(document["pages"])}
                    result.raw_json = {"pages": list(raw_pages)}
                    result.edited_json = None
                    job.updated_at = now_toronto()

                    session.commit()
                    logger.info("Job %s page %s/%s complete in %.1fs", job_id, page_index + 1, document["page_count"], page["elapsed_seconds"])

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
