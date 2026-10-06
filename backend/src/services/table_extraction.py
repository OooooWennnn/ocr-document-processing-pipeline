"""PaddleOCR inference, table reconstruction, text mapping and targeted recognition."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import logging
import os
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from src.services.ocr_limits import detection_side_limit, recognition_batch_size
from src.schemas.schema import BBox, Cell, Table, TableRow, TextItem

MIN_CELL_COVERAGE = 0.85
MIN_CELL_COVERAGE_GAP = 0.15


def box_from(value) -> BBox:
    """Convert model coordinates to BBox; reject non-finite or non-positive bounds."""
    values = value.tolist() if hasattr(value, "tolist") else value

    if len(values) == 4 and not isinstance(values[0], (list, tuple)):
        x1, y1, x2, y2 = map(float, values)
    else:
        x1, y1 = min(p[0] for p in values), min(p[1] for p in values)
        x2, y2 = max(p[0] for p in values), max(p[1] for p in values)

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid OCR bounding box.")

    return BBox(x1, y1, x2, y2)


def coverage(text: BBox, cell: BBox) -> float:
    """Return the fraction of text area inside a cell, not a correctness probability."""
    overlap_width = max(0, min(text.x_max, cell.x_max) - max(text.x_min, cell.x_min))
    overlap_height = max(0, min(text.y_max, cell.y_max) - max(text.y_min, cell.y_min))
    intersection = overlap_width * overlap_height
    text_area = (text.x_max - text.x_min) * (text.y_max - text.y_min)

    return intersection / text_area


def ordered_text(texts: list[TextItem]) -> str:
    """Group overlapping tokens into lines, then join top-to-bottom and left-to-right."""

    lines = []

    for text in sorted(texts, key=lambda t: (t.bbox.y_min, t.bbox.x_min)):
        best = None
        best_overlap = 0.0

        for line in lines:
            anchor = line[0].bbox
            overlap = max(0, min(anchor.y_max, text.bbox.y_max) - max(anchor.y_min, text.bbox.y_min))
            ratio = overlap / min(anchor.y_max - anchor.y_min, text.bbox.y_max - text.bbox.y_min)

            if ratio >= 0.5 and ratio > best_overlap:
                best, best_overlap = line, ratio

        if best is None:
            lines.append([text])
        else:
            best.append(text)

    lines.sort(key=lambda line: min(t.bbox.y_min for t in line))

    return "\n".join(" ".join(t.value for t in sorted(line, key=lambda t: t.bbox.x_min)) for line in lines)


class TableExtraction:
    """Reusable OCR model; detect_document is current and detect supports single-table results."""
    def __init__(self):
        """Initialize OCR once per worker child and share its recognizer with table extraction."""
        from paddleocr import PPStructureV3

        # Keep page coordinates aligned with the source preview.
        # Pages should be upright; text-line orientation remains enabled.
        from src.services.paddlex_compat import without_unused_chart

        with without_unused_chart():
            model_options = {
                "lang": os.getenv("OCR_LANGUAGE", "korean"),
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "use_formula_recognition": False,
                "use_chart_recognition": False,
                "use_region_detection": False,
                "text_det_limit_side_len": int(os.getenv("OCR_DETECTION_SIZE", "2048")),
                "text_det_limit_type": "max",
                "text_rec_score_thresh": 0.0,
                "text_recognition_batch_size": recognition_batch_size(),
                "cpu_threads": int(os.getenv("OCR_CPU_THREADS", "4")),
            }
            self.model = PPStructureV3(**model_options)

        self.share_table_ocr(self.model.paddlex_pipeline)

    @staticmethod
    def share_table_ocr(pipeline):
        """Attach existing OCR when PaddleX 3.4.1 skips table OCR initialization.
        This depends on internal attributes and avoids a second recognizer."""

        layout = getattr(pipeline, "_pipeline", pipeline)
        table_wrapper = getattr(layout, "table_recognition_pipeline", None)
        table = getattr(table_wrapper, "_pipeline", table_wrapper)

        if table is not None and getattr(table, "general_ocr_pipeline", None) is None:
            table.general_ocr_pipeline = layout.general_ocr_pipeline

    def detect(self, image: np.ndarray):
        """Return (raw output, table) for exactly one table, using original image coordinates."""
        import cv2

        target = int(os.getenv("OCR_UPSCALE_TARGET", "1600"))
        scale = min(2.0, max(1.0, target / max(image.shape[:2])))
        prepared = image

        if scale > 1:
            prepared = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        scale_x, scale_y = (prepared.shape[1] / image.shape[1], prepared.shape[0] / image.shape[0])
        predict_options = {
            "use_table_orientation_classify": False,
            "use_ocr_results_with_table_cells": True,
            "text_det_limit_side_len": detection_side_limit(prepared.shape),
            "text_det_limit_type": "max",
        }
        results = list(self.model.predict(prepared, **predict_options))

        if not results:
            raise ValueError("OCR returned no results. Try a clearer image.")

        data = results[0]

        self.validate_table_count(data)

        cells = self.extract_table_structure(data)
        texts = self.extract_texts(data)

        for item in [*cells, *texts]:
            bbox = item.bbox
            item.bbox = BBox(bbox.x_min / scale_x, bbox.y_min / scale_y, bbox.x_max / scale_x, bbox.y_max / scale_y)

        result = self.reconstruct_table(cells, texts)

        if result is None:
            raise ValueError("No usable table cells were detected. Try a clearer image.")

        result["image_width"], result["image_height"] = int(image.shape[1]), int(image.shape[0])
        raw = data.json

        if isinstance(raw, str):
            raw = json.loads(raw)

        raw = {"ocr_scale_x": scale_x, "ocr_scale_y": scale_y, "model_result": raw}

        return raw, result

    def detect_document(self, image, native_texts=None):
        """Return (raw output, page) with all text and detected tables in source coordinates.
        Use native PDF text when available; keep page text if table reconstruction fails."""
        import cv2
        import time

        started = time.perf_counter()
        target = int(os.getenv("OCR_UPSCALE_TARGET", "1600"))
        scale = min(2.0, max(1.0, target / max(image.shape[:2])))
        prepared = image

        if scale > 1:
            prepared = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


        height, width = prepared.shape[:2]
        detector_side = detection_side_limit(prepared.shape)
        batch_size = recognition_batch_size()
        logger = logging.getLogger(__name__)
        logger.info("Page OCR input=%sx%s detector_side=%s recognition_batch=%s", width, height, detector_side, batch_size)

        scale_x, scale_y = (prepared.shape[1] / image.shape[1], prepared.shape[0] / image.shape[0])

        from src.services.native_pdf_ocr import use_native_text

        with use_native_text(self.model, native_texts, scale_x, scale_y):
            predict_options = {
                "use_table_orientation_classify": False,
                "use_ocr_results_with_table_cells": True,
                "text_det_limit_side_len": detection_side_limit(prepared.shape),
                "text_det_limit_type": "max",
            }
            results = list(self.model.predict(prepared, **predict_options))

        if not results:
            raise ValueError("OCR returned no page results.")

        data = results[0]

        def restore(items):
            """Scale item bounds back to source coordinates in place and return the items."""
            for item in items:
                bbox = item.bbox
                item.bbox = BBox(bbox.x_min / scale_x, bbox.y_min / scale_y, bbox.x_max / scale_x, bbox.y_max / scale_y)

            return items

        # Keep full-page text so titles, addresses and notes survive table extraction.
        texts = restore(self.extract_texts({"overall_ocr_res": data.get("overall_ocr_res", {})}))

        if native_texts:
            for text in texts:
                text.score = None

        texts.sort(key=lambda t: (t.bbox.y_min, t.bbox.x_min))

        tables, warnings = [], []

        for index, raw_table in enumerate(data.get("table_res_list", [])):
            try:
                cells = restore(self.extract_cells(raw_table.get("cell_box_list", [])))

                from src.services.cell_geometry import recover_ruled_cells, refine_cell_geometry

                tokens = restore(self.extract_texts({"overall_ocr_res": data.get("overall_ocr_res", {}), "table_res_list": [raw_table]}))

                if native_texts:
                    for token in tokens:
                        token.score = None

                try:
                    refined_cells = refine_cell_geometry(image, cells)
                    table = self.reconstruct_table(refined_cells, tokens)

                    if table is None:
                        raise ValueError("No usable cells were detected.")
                except ValueError:
                    # Try source rules before discarding an invalid table.
                    recovered = recover_ruled_cells(image, cells)

                    if not recovered:
                        raise

                    table = self.reconstruct_table(recovered, tokens)
                    message = f"Table {index + 1}: Cell structure recovered from source lines. Review the table boundaries."

                    table["warnings"].append(message)
                    warnings.append(message)

                table.update(id=str(index + 1), image_width=int(image.shape[1]), image_height=int(image.shape[0]))
                tables.append(table)
            except ValueError as exc:
                warnings.append(f"Table {index + 1}: {exc} Full-page text is still available.")

        refinement = {"attempted_crops": 0, "accepted_lines": 0}

        if not native_texts and hasattr(self.model, "paddlex_pipeline"):
            # Re-read selected crops with the existing recognizer.
            from src.services.ocr_refinement import refine_page

            layout = getattr(self.model.paddlex_pipeline, "_pipeline", self.model.paddlex_pipeline)

            def recognize(crops):
                """Read selected crops with the already loaded recognizer."""
                return layout.general_ocr_pipeline.text_rec_model(crops, batch_size=recognition_batch_size())

            try:
                texts, refinement = refine_page(image, texts, tables, recognize)
            except (ValueError, RuntimeError) as exc:
                warnings.append(f"Targeted OCR refinement unavailable: {exc}")

            texts.sort(key=lambda t: (t.bbox.y_min, t.bbox.x_min))

        raw = data.json

        if isinstance(raw, str):
            raw = json.loads(raw)

        text_source = "pdf_text" if native_texts else "ocr"
        serialized_texts = []

        for index, text in enumerate(texts):
            item = asdict(text)
            item.update(id=str(index), original_value=text.value, source=text_source)
            serialized_texts.append(item)

        raw_result = {
            "ocr_scale_x": scale_x,
            "ocr_scale_y": scale_y,
            "model_result": raw,
        }
        page_result = {
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "texts": serialized_texts,
            "tables": tables,
            "warnings": warnings,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "text_source": text_source,
            "native_text_fast_path": bool(native_texts),
            "refinement": refinement,
        }

        return raw_result, page_result

    @staticmethod
    def validate_table_count(data):
        """Require exactly one table in the legacy single-table path."""
        if len(data.get("table_res_list", [])) != 1:
            raise ValueError("The image must contain exactly one detected table.")

    def extract_texts(self, data: dict) -> list[TextItem]:
        """Return TextItems, preferring cell OCR when available.
        Preserve texts with unknown scores; reject missing coordinates."""
        overall = data.get("overall_ocr_res", {})
        tables = data.get("table_res_list", [])
        table_ocr = tables[0].get("table_ocr_pred", {}) if tables else {}
        # Cell OCR can split lines that cross field boundaries.
        # Do not zip texts with scores: shorter score lists would drop text.
        table_values = table_ocr.get("rec_texts", [])
        table_boxes = table_ocr.get("rec_boxes", [])
        has_cell_ocr = len(table_values) > 0 and len(table_boxes) == len(table_values)
        ocr = table_ocr if has_cell_ocr else overall
        values = ocr.get("rec_texts", [])
        boxes = ocr.get("rec_boxes", ocr.get("rec_polys", []))
        scores = ocr.get("rec_scores", [])
        aligned = len(scores) == len(values) == len(boxes)
        items = []

        for i, value in enumerate(values):
            if i >= len(boxes):
                raise ValueError("OCR text and coordinates are inconsistent. Please retry.")

            box = box_from(boxes[i])
            score = float(scores[i]) if aligned and scores[i] is not None else None

            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                score = None

            if not aligned:
                # Recover scores only from matching text and position.
                # Keep None when a split token has no reliable score.
                candidates = []

                for j, original in enumerate(overall.get("rec_texts", [])):
                    if original != value:
                        continue

                    if j >= len(overall.get("rec_boxes", [])) or j >= len(overall.get("rec_scores", [])):
                        continue

                    original_box = box_from(overall["rec_boxes"][j])
                    overlap = min(coverage(box, original_box), coverage(original_box, box))

                    if overlap >= 0.8:
                        candidate = overall["rec_scores"][j]

                        if candidate is not None:
                            candidate = float(candidate)

                            if math.isfinite(candidate) and 0 <= candidate <= 1:
                                candidates.append(candidate)

                score = min(candidates) if candidates else None

            if str(value).strip():
                items.append(TextItem(str(value), score, box))

        return items

    def extract_cells(self, boxes: list) -> list[Cell]:
        """Convert model boxes to cells without assigned grid positions."""
        return [Cell(-1, -1, box_from(box)) for box in boxes]

    def extract_table_structure(self, data: dict) -> list[Cell]:
        """Get first-table cells for the legacy single-table path."""
        return self.extract_cells(data["table_res_list"][0].get("cell_box_list", []))

    def reconstruct_table(self, cells: list[Cell], texts: list[TextItem]):
        """Assign grid positions and map text; reject overlapping or unusable cells.
        Return the table with original text, bounds and unmapped tokens."""
        if not cells:
            return None

        x_lines, y_lines = self.get_grid_boundaries(cells)

        self.assign_grid_position(cells, x_lines, y_lines)

        occupied = set()

        for cell in cells:
            if cell.row < 0 or cell.col < 0 or cell.rowspan < 1 or cell.colspan < 1:
                raise ValueError("Some cell boundaries could not be resolved. Try a clearer image.")

            for row in range(cell.row, cell.row + cell.rowspan):
                for col in range(cell.col, cell.col + cell.colspan):
                    if (row, col) in occupied:
                        raise ValueError("Detected cells overlap. Try a clearer image.")

                    occupied.add((row, col))

        unmapped = self.assign_texts_to_cells(cells, texts)
        table = Table("1", BBox(x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]), [])

        for row in range(max(c.row + c.rowspan for c in cells)):
            row_box = BBox(x_lines[0], y_lines[row], x_lines[-1], y_lines[row + 1])
            row_cells = sorted((cell for cell in cells if cell.row == row), key=lambda cell: cell.col)
            table.table_rows.append(TableRow(row, row_box, row_cells))

        result = asdict(table)

        result.update(x_lines=x_lines, y_lines=y_lines, unmapped_texts=[asdict(t) for t in unmapped], mapping_version=2)

        for row in result["table_rows"]:
            for cell in row["cells"]:
                original = next(c for c in cells if c.row == cell["row"] and c.col == cell["col"])
                cell["original_value"] = ordered_text(original.texts)
                cell["mapping_score"] = getattr(original, "mapping_score", None)
                cell["mapping_warning"] = getattr(original, "mapping_warning", False)

        result["warnings"] = ["Some text could not be assigned confidently. Review highlighted cells and unmapped text."] if unmapped else []

        return result

    def assign_texts_to_cells(self, cells, texts):
        """Update cells only for 85% coverage and a 15-percentage-point lead over the next match.
        Return ambiguous text and mark candidate cells for review."""
        for cell in cells:
            cell.texts = []
            cell.mapping_score = None
            cell.mapping_warning = False

        left = min(cell.bbox.x_min for cell in cells)
        top = min(cell.bbox.y_min for cell in cells)
        right = max(cell.bbox.x_max for cell in cells)
        bottom = max(cell.bbox.y_max for cell in cells)
        bounds = BBox(left, top, right, bottom)
        unmapped = []

        for text in texts:
            if coverage(text.bbox, bounds) == 0:
                continue  # Keep outside captions separate from unmapped table text.

            ranked = sorted(((coverage(text.bbox, c.bbox), i) for i, c in enumerate(cells)), reverse=True)
            best, index = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0

            if best < MIN_CELL_COVERAGE or best - second < MIN_CELL_COVERAGE_GAP:
                unmapped.append(text)

                for score, i in ranked:
                    if score > 0.1:
                        cells[i].mapping_warning = True

                continue

            cell = cells[index]

            cell.texts.append(text)

            cell.mapping_score = min(cell.mapping_score, best) if cell.mapping_score is not None else best

        for cell in cells:
            cell.texts.sort(key=lambda t: (t.bbox.y_min, t.bbox.x_min))

        return unmapped

    def get_grid_boundaries(self, cells):
        """Cluster cell edges using a tolerance based on the narrowest cell."""
        x_tolerance = min(10.0, min(c.bbox.x_max - c.bbox.x_min for c in cells) * 0.15)
        y_tolerance = min(10.0, min(c.bbox.y_max - c.bbox.y_min for c in cells) * 0.15)

        x_edges = [value for cell in cells for value in (cell.bbox.x_min, cell.bbox.x_max)]
        y_edges = [value for cell in cells for value in (cell.bbox.y_min, cell.bbox.y_max)]
        x_lines = self.merge_lines(x_edges, x_tolerance)
        y_lines = self.merge_lines(y_edges, y_tolerance)

        return x_lines, y_lines

    @staticmethod
    def merge_lines(lines, tolerance=10):
        """Represent nearby edges by their median; limit group width to avoid chaining."""
        clusters = []

        for value in sorted(lines):
            if clusters and value - clusters[-1][0] <= 2 * tolerance and value - median(clusters[-1]) <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        return [float(median(cluster)) for cluster in clusters]

    @staticmethod
    def find_index(lines, value):
        """Return the nearest grid-line index, or None for an empty list."""
        return min(range(len(lines)), key=lambda i: abs(lines[i] - value)) if lines else None

    def assign_grid_position(self, cells, x_lines, y_lines):
        """Assign row, column and spans to cells in place using the grid boundaries."""
        for cell in cells:
            col, col_end = self.find_index(x_lines, cell.bbox.x_min), self.find_index(x_lines, cell.bbox.x_max)
            row, row_end = self.find_index(y_lines, cell.bbox.y_min), self.find_index(y_lines, cell.bbox.y_max)

            if None in (col, col_end, row, row_end) or col_end <= col or row_end <= row:
                continue

            cell.row, cell.col = row, col
            cell.rowspan, cell.colspan = row_end - row, col_end - col
