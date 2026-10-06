"""Initialize and reuse one OCR model per worker process."""

import logging
import time
from threading import Lock
from src.services.table_extraction import TableExtraction

logger = logging.getLogger(__name__)
_extractor = None
_lock = Lock()


def get_extractor():
    """Return this process's OCR model, creating it once under a lock."""
    global _extractor

    with _lock:
        if _extractor is None:
            started = time.perf_counter()

            logger.info("Loading document OCR model")

            _extractor = TableExtraction()

            logger.info("Document OCR model ready in %.1fs", time.perf_counter() - started)

        return _extractor
