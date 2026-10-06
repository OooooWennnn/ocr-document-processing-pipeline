"""Combine text and cells into one positioned, editable document."""

from statistics import median
import re
from src.schemas.schema import BBox, TextItem
from src.services.table_extraction import MIN_CELL_COVERAGE, MIN_CELL_COVERAGE_GAP, ordered_text


def intersection(a, b):
    """Return the overlapping rectangle, or None."""
    x1, y1 = max(a["x_min"], b["x_min"]), max(a["y_min"], b["y_min"])
    x2, y2 = min(a["x_max"], b["x_max"]), min(a["y_max"], b["y_max"])

    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def covered_fraction(box, boxes):
    """Measure covered area without counting overlapping boxes twice."""
    intersections = [overlap for other in boxes if (overlap := intersection(box, other))]
    area = (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])

    if not intersections or area <= 0:
        return 0.0

    x_edges = sorted({x for overlap in intersections for x in (overlap[0], overlap[2])})
    covered = 0.0

    for left, right in zip(x_edges, x_edges[1:]):
        intervals = sorted((overlap[1], overlap[3]) for overlap in intersections if overlap[0] <= left and overlap[2] >= right)
        length, end = 0.0, float("-inf")

        for bottom, top in intervals:
            length += max(0, top - max(end, bottom))
            end = max(end, top)

        covered += (right - left) * length

    return min(1.0, covered / area)


def normalize(value):
    """Ignore whitespace and case when comparing OCR text."""
    return re.sub(r"\s+", "", value).casefold()


def font_size(box, tokens=None, value=""):
    """Estimate display size from token or region height, not the original font."""
    heights = [token["bbox"]["y_max"] - token["bbox"]["y_min"] for token in (tokens or [])]

    if heights:
        height = median(heights)
    else:
        line_count = max(1, len(value.splitlines()))
        height = (box["y_max"] - box["y_min"]) / line_count

    return round(max(6.0, min(48.0, height * 0.8)), 2)


def join_tokens(tokens, key):
    """Join a text field in line and word order."""
    texts = [TextItem(token[key], token.get("score"), BBox(**token["bbox"])) for token in tokens]

    return ordered_text(texts)


def present_page(page, tables):
    """Return positioned regions without mutating input text or cells.
    Preserve token coordinates and scores, keeping ambiguous text outside cells."""
    tables = [{**table, "cells": [dict(cell) for cell in table["cells"]]} for table in tables]
    raw_cells = {}

    for table in page.get("tables", []):
        for row in table.get("table_rows", []):
            for cell in row["cells"]:
                key = (str(table["id"]), cell["row"], cell["col"])
                raw_cells[key] = cell

    cells = [(table, cell) for table in tables for cell in table["cells"]]
    texts = []

    for text in page.get("texts", []):
        original_value = text.get("original_value", text["value"])
        visible_text = {
            **text,
            "value": text.get("edited_value", original_value),
            "original_value": original_value,
            "edited": "edited_value" in text,
        }
        texts.append(visible_text)

    cell_texts = {}
    duplicate_text_ids = set()

    for text in texts:
        overlaps = ((covered_fraction(text["bbox"], [cell["bbox"]]), index) for index, (_, cell) in enumerate(cells))
        ranked = sorted(overlaps, reverse=True)

        if not ranked:
            continue

        best_coverage, cell_index = ranked[0]
        clear_match = len(ranked) == 1 or best_coverage - ranked[1][0] >= MIN_CELL_COVERAGE_GAP

        if best_coverage >= MIN_CELL_COVERAGE and clear_match:
            cell_texts.setdefault(cell_index, []).append(text)
            continue

        filled_boxes = [cell["bbox"] for _, cell in cells if cell.get("original_value") or cell["value"]]

        if covered_fraction(text["bbox"], filled_boxes) >= 0.9:
            # Match text as well as position before removing a duplicate.
            candidates = [cell for _, cell in cells if intersection(text["bbox"], cell["bbox"]) and cell.get("original_value")]
            candidates = sorted(candidates, key=lambda cell: (cell["bbox"]["y_min"], cell["bbox"]["x_min"]))
            normalized_text = normalize(text["original_value"])
            combined_text = " ".join(cell["original_value"] for cell in candidates)
            normalized_cell_text = normalize(combined_text)

            if normalized_text and normalized_text in normalized_cell_text:
                duplicate_text_ids.add(text["id"])

    owned_text_ids = set(duplicate_text_ids)
    regions = []

    for index, (table, cell) in enumerate(cells):
        linked_texts = sorted(cell_texts.get(index, []), key=lambda text: (text["bbox"]["y_min"], text["bbox"]["x_min"]))
        owned_text_ids.update(text["id"] for text in linked_texts)

        key = (str(table["id"]), cell["row"], cell["col"])
        raw_cell = raw_cells.get(key, {})

        # Fill missed cell text only from unambiguous full-page matches.
        if not cell.get("original_value") and linked_texts:
            cell["score"] = None

            if all(text.get("score") is not None for text in linked_texts):
                cell["score"] = min((text["score"] for text in linked_texts if text.get("score") is not None), default=None)

            cell["mapping_score"] = min(covered_fraction(text["bbox"], [cell["bbox"]]) for text in linked_texts)
            cell["original_value"] = join_tokens(linked_texts, "original_value")

            if not cell.get("edited"):
                cell["value"] = join_tokens(linked_texts, "value")
        elif linked_texts and any(text["edited"] for text in linked_texts) and not cell.get("edited"):
            # Preserve old text edits, giving saved cell edits priority.
            cell["value"] = join_tokens(linked_texts, "value")
            cell["edited"] = True

        tokens = sorted(raw_cell.get("texts", []) or linked_texts, key=lambda text: (text["bbox"]["y_min"], text["bbox"]["x_min"]))
        size = font_size(cell["bbox"], tokens, cell.get("original_value", "")) if tokens else 12.0
        visible_tokens = [{**token, "font_size": font_size(token["bbox"], value=token["value"])} for token in tokens]
        region = {
            **cell,
            "id": f"cell:{table['id']}:{cell['row']}:{cell['col']}",
            "kind": "cell",
            "table_id": str(table["id"]),
            "font_size": size,
            "source_text_ids": [text["id"] for text in linked_texts],
            "tokens": visible_tokens,
            "recognition_refined": raw_cell.get("recognition_refined", False),
        }
        regions.append(region)

    standalone_texts = [text for text in texts if text["id"] not in owned_text_ids]

    for text in standalone_texts:
        overlapping_cells = sum(covered_fraction(text["bbox"], [cell["bbox"]]) > 0.1 for _, cell in cells)

        if overlapping_cells > 1:
            text["mapping_warning"] = True

    for text in standalone_texts:
        region = {
            **text,
            "id": f"text:{text['id']}",
            "text_id": text["id"],
            "kind": "text",
            "font_size": font_size(text["bbox"], value=text["original_value"]),
        }
        regions.append(region)

    regions.sort(key=lambda region: (region["bbox"]["y_min"], region["bbox"]["x_min"], region["kind"], region["id"]))

    return {
        **page,
        "texts": standalone_texts,
        "tables": tables,
        "regions": regions,
        "layout_version": 1,
    }
