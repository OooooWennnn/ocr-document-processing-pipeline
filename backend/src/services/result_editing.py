"""Present stored results and validate document edits."""

from copy import deepcopy


def apply_edits(original: dict, edits: list) -> dict:
    """Return a copy with updated cell values; preserve the original table."""
    result = deepcopy(original)
    cells = {(cell["row"], cell["col"]): cell for row in result["table_rows"] for cell in row["cells"]}
    seen = set()

    for edit in edits:
        key = (edit.row, edit.col)

        if key not in cells or key in seen:
            raise ValueError("The edit contains an unknown or duplicate cell.")

        seen.add(key)
        cells[key]["edited_value"] = edit.value

    return result


def present_table(result: dict) -> dict:
    """Flatten saved rows for the client, including edits, originals and scores."""
    if "x_lines" not in result or "y_lines" not in result:
        from src.services.table_extraction import TableExtraction

        # Reuse extraction boundaries when presenting legacy results.
        raw_cells = [cell for row in result["table_rows"] for cell in row["cells"]]
        x_edges = [cell["bbox"][key] for cell in raw_cells for key in ("x_min", "x_max")]
        y_edges = [cell["bbox"][key] for cell in raw_cells for key in ("y_min", "y_max")]
        result = {
            **result,
            "x_lines": TableExtraction.merge_lines(x_edges),
            "y_lines": TableExtraction.merge_lines(y_edges),
        }

    cells = []

    for row in result["table_rows"]:
        for cell in row["cells"]:
            text_value = " ".join(text["value"] for text in cell["texts"])
            original_value = cell.get("original_value", text_value)
            value = cell.get("edited_value", original_value)
            score = None

            # Unknown token scores leave cell confidence unknown.
            if all(text.get("score") is not None for text in cell["texts"]):
                score = min((text["score"] for text in cell["texts"] if text.get("score") is not None), default=None)

            visible_cell = {
                **{key: cell[key] for key in ("row", "col", "rowspan", "colspan", "bbox")},
                "value": value,
                "original_value": original_value,
                "edited": "edited_value" in cell,
                "mapping_score": cell.get("mapping_score"),
                "mapping_warning": cell.get("mapping_warning", False),
                "score": score,
            }
            cells.append(visible_cell)

    return {
        "id": str(result["id"]),
        "bbox": result["bbox"],
        "x_lines": result["x_lines"],
        "y_lines": result["y_lines"],
        "warnings": result.get("warnings", []),
        "unmapped_texts": result.get("unmapped_texts", []),
        "image_width": result.get("image_width"),
        "image_height": result.get("image_height"),
        "cells": cells,
    }


def present_document(result):
    """Return pages and regions, including support for older single-table results."""
    from src.services.document_layout import present_page

    if result.get("schema_version") != 3:
        page = {
            "page_index": 0,
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "texts": [],
            "tables": [result],
            "warnings": ["Legacy table-only result. Upload again for full-document extraction."],
        }
        pages = [present_page(page, [present_table(result)])]

        return {
            "pages": pages,
            "tables": pages[0]["tables"],
            "page_count": 1,
            "completed_pages": 1,
            "layout_version": 1,
        }

    pages = []

    for page in result["pages"]:
        tables = [present_table(table) for table in page["tables"]]
        pages.append(present_page(page, tables))

    return {
        "pages": pages,
        "tables": pages[0]["tables"] if pages else [],
        "page_count": result["page_count"],
        "completed_pages": len(pages),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "layout_version": 1,
    }


def apply_regions(result, page_index, edits):
    """Apply region edits to the supplied result in place; callers must pass a copy."""
    visible_pages = present_document(result)["pages"]
    page = next((page for page in visible_pages if page["page_index"] == page_index), None)

    if page is None:
        raise ValueError("Unknown page.")

    regions = {region["id"]: region for region in page["regions"]}
    seen = set()
    raw_page = None

    if result.get("schema_version") == 3:
        raw_page = next((page for page in result["pages"] if page["page_index"] == page_index), None)

    for edit in edits:
        if edit.id not in regions or edit.id in seen:
            raise ValueError("Unknown or duplicate document region.")

        seen.add(edit.id)
        region = regions[edit.id]

        if region["kind"] == "cell":
            table = result

            if raw_page is not None:
                table = next(table for table in raw_page["tables"] if str(table["id"]) == region["table_id"])

            table_cells = (cell for row in table["table_rows"] for cell in row["cells"])
            cell = next(cell for cell in table_cells if cell["row"] == region["row"] and cell["col"] == region["col"])
            cell["edited_value"] = edit.value
        else:
            text = next(text for text in raw_page["texts"] if text["id"] == region["text_id"])
            text["edited_value"] = edit.value

    return result


def apply_document_edits(original, edit):
    """Validate edits and return an edited copy; the API handles database persistence."""
    result = deepcopy(original)
    pages = getattr(edit, "pages", [])
    regions = getattr(edit, "regions", [])

    if pages or regions:
        if (pages and regions) or edit.cells or edit.texts:
            raise ValueError("Use one unified document edit format per request.")

        updates = pages or [edit]
        indices = set()

        for update in updates:
            if update.page_index in indices:
                raise ValueError("Duplicate page edits.")

            indices.add(update.page_index)
            apply_regions(result, update.page_index, update.regions)

        return result

    if original.get("schema_version") != 3:
        if edit.page_index != 0 or edit.table_id not in (None, str(original["id"])) or edit.texts:
            raise ValueError("Unknown page or table.")

        return apply_edits(result, edit.cells)

    page = next((page for page in result["pages"] if page["page_index"] == edit.page_index), None)

    if page is None:
        raise ValueError("Unknown page.")

    if edit.cells:
        table = next((table for table in page["tables"] if str(table["id"]) == edit.table_id), None)

        if table is None:
            raise ValueError("Unknown table. Supply its table_id.")

        index = page["tables"].index(table)
        page["tables"][index] = apply_edits(table, edit.cells)

    # Reject text edits that would duplicate an existing cell edit.
    visible = next(page for page in present_document(result)["pages"] if page["page_index"] == edit.page_index)
    allowed = {text["id"] for text in visible["texts"]}
    texts = {text["id"]: text for text in page["texts"]}
    seen = set()

    for text in edit.texts:
        if text.id not in allowed or text.id in seen:
            raise ValueError("Unknown, duplicate or cell-owned text region. Edit the document cell instead.")

        seen.add(text.id)
        texts[text.id]["edited_value"] = text.value

    return result
