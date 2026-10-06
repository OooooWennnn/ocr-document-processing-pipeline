"""Bound text detection size and recognition batches to limit memory use."""

import math
import os


def recognition_batch_size():
    """Return the batch size, clamped to 1-16; default is 4."""
    return max(1, min(16, int(os.getenv("OCR_RECOGNITION_BATCH_SIZE", "4"))))


def detection_side_limit(shape):
    """Fit the detector within the pixel budget, including 32px alignment.
    Recognition crops stay unchanged; a smaller limit can miss small text."""
    height, width = shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions.")

    configured = max(32, int(os.getenv("OCR_DETECTION_SIZE", "2048")))
    budget = max(1024, int(os.getenv("OCR_DETECTION_MAX_PIXELS", "1000000")))
    longer, shorter = max(height, width), min(height, width)
    limit = min(configured, longer, int(math.sqrt(budget * longer / shorter)))
    limit = max(32, limit // 32 * 32)

    # Include rounding of the short side in the 32px-aligned pixel budget.
    while limit > 32 and limit * (math.ceil(shorter * limit / longer / 32) * 32) > budget:
        limit -= 32

    return limit
