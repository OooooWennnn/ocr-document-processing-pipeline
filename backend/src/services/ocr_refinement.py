"""Re-read low-confidence or boundary-crossing text within a crop budget."""

from dataclasses import asdict
import math
import os
from src.schemas.schema import BBox, TextItem
from src.services.table_extraction import MIN_CELL_COVERAGE, coverage, ordered_text

LOW_CONFIDENCE_SCORE = 0.85
MIN_REFINED_SCORE = 0.80
MIN_CROSSING_SCORE = 0.75
MIN_SCORE_GAIN = 0.04


def refine_page(image, texts, tables, recognize):
    """Return accepted text and refinement stats; update matching table cells in place.
    Reuse the recognizer and leave the source image unchanged."""
    import cv2

    cells = [c for t in tables for row in t["table_rows"] for c in row["cells"]]
    crop_requests = []
    line_groups = []
    scheduled_indices = set()

    # Remove long rules from a copy so they do not interfere with text crops.
    ink = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, image.shape[1] // 25), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, image.shape[0] // 25)))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vertical_kernel)
    rules = cv2.dilate(cv2.bitwise_or(horizontal, vertical), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cleaned = image.copy()
    cleaned[rules > 0] = 255
    limit = max(0, min(128, int(os.getenv("OCR_REFINE_MAX_CROPS", "64"))))

    for text_index, text in enumerate(texts):
        if text_index in scheduled_indices:
            continue

        ranked = sorted([(coverage(text.bbox, BBox(**c["bbox"])), cell_index) for cell_index, c in enumerate(cells)], reverse=True)
        crossing = ranked and ranked[0][0] < MIN_CELL_COVERAGE and sum(overlap_score > 0.1 for overlap_score, _ in ranked) > 1

        if not crossing and (text.score is None or text.score >= LOW_CONFIDENCE_SCORE):
            continue

        # Include nearby text to give short, uncertain fragments word context.
        line_indices = [text_index]

        if not crossing:
            owner = ranked[0][1] if ranked and ranked[0][0] >= MIN_CELL_COVERAGE else None

            for neighbor_index, other in enumerate(texts):
                if neighbor_index == text_index or neighbor_index in scheduled_indices:
                    continue

                overlap = max(0, min(text.bbox.y_max, other.bbox.y_max) - max(text.bbox.y_min, other.bbox.y_min))
                height = min(text.bbox.y_max - text.bbox.y_min, other.bbox.y_max - other.bbox.y_min)
                gap = max(0, max(text.bbox.x_min, other.bbox.x_min) - min(text.bbox.x_max, other.bbox.x_max))

                if not (overlap / height >= 0.6 and gap <= height * 2):
                    continue

                if owner is None or coverage(other.bbox, BBox(**cells[owner]["bbox"])) >= MIN_CELL_COVERAGE:
                    line_indices.append(neighbor_index)

        left = min(texts[index].bbox.x_min for index in line_indices)
        top = min(texts[index].bbox.y_min for index in line_indices)
        right = max(texts[index].bbox.x_max for index in line_indices)
        bottom = max(texts[index].bbox.y_max for index in line_indices)
        line_box = BBox(left, top, right, bottom)
        candidates = []

        if crossing:
            for score, cell_index in sorted(ranked, key=lambda pair: cells[pair[1]]["bbox"]["x_min"]):
                if score <= 0.1:
                    continue

                cell_box = BBox(**cells[cell_index]["bbox"])

                left = max(line_box.x_min, cell_box.x_min + 1)
                top = max(line_box.y_min, cell_box.y_min + 1)
                right = min(line_box.x_max, cell_box.x_max - 1)
                bottom = min(line_box.y_max, cell_box.y_max - 1)
                crop_box = BBox(left, top, right, bottom)
                candidates.append((crop_box, cell_index))
        else:
            owner = ranked[0][1] if ranked and ranked[0][0] >= MIN_CELL_COVERAGE else None
            candidates = [(line_box, owner)]

        if len(crop_requests) + len(candidates) > limit:
            continue

        crop_indices = []

        for box, owner in candidates:
            x1 = max(0, math.floor(box.x_min))
            x2 = min(image.shape[1], math.ceil(box.x_max))
            y1 = max(0, math.floor(box.y_min) - 1)
            y2 = min(image.shape[0], math.ceil(box.y_max) + 1)

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            crop = cleaned[y1:y2, x1:x2]
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            crop = cv2.copyMakeBorder(crop, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            crop_indices.append(len(crop_requests))
            crop_requests.append((crop, box, owner))

        if crossing and sum(coverage(line_box, crop_requests[request_index][1]) for request_index in crop_indices) < 0.85:
            continue

        if crop_indices:
            line_groups.append((line_indices, crop_indices, bool(crossing)))
            scheduled_indices.update(line_indices)

    if not crop_requests:
        return texts, {"attempted_crops": 0, "accepted_lines": 0}

    predictions = list(recognize([r[0] for r in crop_requests]))

    if len(predictions) != len(crop_requests):
        raise ValueError("Refinement recognition results are not aligned.")

    replaced_indices = set()
    refined_texts = []
    accepted_lines = 0

    for line_indices, crop_indices, crossing in line_groups:
        candidates = []

        for request_index in crop_indices:
            prediction = predictions[request_index]
            value = str(prediction.get("rec_text", "")).strip()
            score = float(prediction.get("rec_score", 0))

            if not value or not math.isfinite(score) or not 0 <= score <= 1:
                break

            candidates.append(TextItem(value, score, crop_requests[request_index][1]))

        if len(candidates) != len(crop_indices):
            continue

        known_scores = (texts[index].score for index in line_indices if texts[index].score is not None)
        old_score = min(known_scores, default=0)

        # Replace a re-read line only when confidence improves enough.
        # Accept split lines only when every fragment passes the score threshold.
        if crossing:
            if min(text.score for text in candidates) < MIN_CROSSING_SCORE:
                continue
        elif candidates[0].score < max(MIN_REFINED_SCORE, old_score + MIN_SCORE_GAIN):
            continue

        replaced_indices.update(line_indices)
        refined_texts.extend(candidates)

        accepted_lines += 1

        for cell in cells:
            cell_box = BBox(**cell["bbox"])
            matched_texts = [text for text in candidates if coverage(text.bbox, cell_box) >= MIN_CELL_COVERAGE]

            if not matched_texts:
                continue

            existing_texts = [TextItem(t["value"], t.get("score"), BBox(**t["bbox"])) for t in cell["texts"]]
            remaining = []

            for text in existing_texts:
                replaced = False

                for replacement in matched_texts:
                    overlap = max(coverage(text.bbox, replacement.bbox), coverage(replacement.bbox, text.bbox))

                    if overlap >= 0.5:
                        replaced = True
                        break

                if not replaced:
                    remaining.append(text)

            existing_texts = remaining
            combined_texts = [*existing_texts, *matched_texts]
            cell["texts"] = [asdict(text) for text in combined_texts]
            cell["original_value"] = ordered_text(combined_texts)
            cell["mapping_score"] = min(coverage(text.bbox, cell_box) for text in combined_texts)
            cell["mapping_warning"] = bool(crossing or any(text.score is None or text.score < 0.9 for text in combined_texts))
            cell["recognition_refined"] = True

    remaining_texts = [text for index, text in enumerate(texts) if index not in replaced_indices]
    result_texts = remaining_texts + refined_texts
    statistics = {
        "attempted_crops": len(crop_requests),
        "accepted_lines": accepted_lines,
    }

    return result_texts, statistics
