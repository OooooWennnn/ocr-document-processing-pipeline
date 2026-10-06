import unittest
from copy import deepcopy
from pathlib import Path
import json
import test_api_flow as api_tests
from test_api_flow import engine
from sqlmodel import Session
from src.models.job_model import Job, JobResult, JobStatus
from src.schemas.schema import Cell, BBox, TextItem
from src.services.table_extraction import TableExtraction
from src.services.result_editing import present_document
from src.services.document_layout import covered_fraction


def fixture():
    x = TableExtraction.__new__(TableExtraction)
    table = x.reconstruct_table([Cell(-1, -1, BBox(0, 40, 100, 80)), Cell(-1, -1, BBox(100, 40, 200, 80))], [TextItem("Item 1", 0.9, BBox(5, 50, 50, 65)), TextItem("1.00", 0.95, BBox(120, 50, 160, 65))])

    def token(id, value, box):
        return {"id": id, "value": value, "original_value": value, "score": 0.9, "bbox": dict(zip(["x_min", "y_min", "x_max", "y_max"], box))}

    page = {"page_index": 0, "image_width": 200, "image_height": 120, "texts": [token("title", "Packing List", [10, 0, 150, 25]), token("item", "Item 1", [5, 50, 50, 65]), token("quantity", "1.00", [120, 50, 160, 65]), token("notes", "Notes", [0, 90, 60, 110])], "tables": [table]}

    return {"schema_version": 3, "page_count": 2, "pages": [page, {**deepcopy(page), "page_index": 1}]}


class UnifiedDocumentTests(unittest.TestCase):
    setUp = api_tests.ApiFlowTests.setUp
    tearDown = api_tests.ApiFlowTests.tearDown

    def seed(self, document):
        with Session(engine) as session:
            session.add(Job(id=20, original_filename="document.pdf", file_type="application/pdf", status=JobStatus.done, page_count=2))
            session.commit()
            session.add(JobResult(job_id=20, result_json=document))
            session.commit()

        return "/api/v1/jobs/20/result"

    def test_one_owner_per_region_and_exact_source_coordinates(self):
        raw = fixture()
        original = deepcopy(raw)
        page = present_document(raw)["pages"][0]

        self.assertEqual([t["id"] for t in page["texts"]], ["title", "notes"])
        self.assertEqual(len(page["regions"]), 4)
        self.assertEqual(sum(r["value"] == "Item 1" for r in page["regions"]), 1)

        title = next(r for r in page["regions"] if r["id"] == "text:title")

        self.assertEqual(title["bbox"], raw["pages"][0]["texts"][0]["bbox"])

        cell = next(r for r in page["regions"] if r["id"] == "cell:1:0:0")

        self.assertEqual(cell["source_text_ids"], ["item"])
        self.assertEqual(raw, original)

    def test_atomic_save_of_title_and_table_across_pages(self):
        url = self.seed(fixture())
        response = self.client.patch(url, json={"pages": [{"page_index": 0, "regions": [{"id": "text:title", "value": "Updated title"}, {"id": "cell:1:0:0", "value": "Item corrected"}]}, {"page_index": 1, "regions": [{"id": "text:notes", "value": "Second page note"}]}]})

        self.assertEqual(response.status_code, 200, response.text)

        pages = self.client.get(url).json()["pages"]

        self.assertEqual(next(r for r in pages[0]["regions"] if r["id"] == "text:title")["value"], "Updated title")
        self.assertEqual(next(r for r in pages[0]["regions"] if r["id"] == "cell:1:0:0")["value"], "Item corrected")
        self.assertEqual(pages[0]["tables"][0]["cells"][0]["value"], "Item corrected")
        self.assertEqual(next(r for r in pages[1]["regions"] if r["id"] == "text:notes")["value"], "Second page note")
        self.assertEqual(next(r for r in pages[0]["regions"] if r["id"] == "cell:1:0:0")["original_value"], "Item 1")

        bad = self.client.patch(url, json={"pages": [{"page_index": 0, "regions": [{"id": "text:title", "value": "Must rollback"}]}, {"page_index": 1, "regions": [{"id": "missing", "value": "bad"}]}]})

        self.assertEqual(bad.status_code, 422)
        self.assertEqual(next(r for r in self.client.get(url).json()["pages"][0]["regions"] if r["id"] == "text:title")["value"], "Updated title")

    def test_old_text_api_cannot_create_second_cell_edit_and_empty_edit_keeps_ownership(self):
        url = self.seed(fixture())
        response = self.client.patch(url, json={"page_index": 0, "texts": [{"id": "item", "value": "conflicting"}]})

        self.assertEqual(response.status_code, 422)

        response = self.client.patch(url, json={"page_index": 0, "regions": [{"id": "cell:1:0:0", "value": ""}]})

        self.assertEqual(response.status_code, 200, response.text)

        page = response.json()["pages"][0]

        self.assertEqual(next(r for r in page["regions"] if r["id"] == "cell:1:0:0")["value"], "")
        self.assertFalse(any(r.get("text_id") == "item" for r in page["regions"]))

    def test_missing_cell_text_recovered_and_conflicting_boundary_text_retained(self):
        raw = fixture()
        cell = raw["pages"][0]["tables"][0]["table_rows"][0]["cells"][0]
        cell["texts"] = []
        cell["original_value"] = ""
        page = present_document(raw)["pages"][0]

        self.assertEqual(next(r for r in page["regions"] if r["id"] == "cell:1:0:0")["value"], "Item 1")
        raw["pages"][0]["texts"].append({"id": "crossing", "value": "unresolved extra words", "original_value": "unresolved extra words", "score": 0.9, "bbox": {"x_min": 70, "y_min": 50, "x_max": 130, "y_max": 65}})

        page = present_document(raw)["pages"][0]

        self.assertTrue(any(r["id"] == "text:crossing" for r in page["regions"]))

    def test_merged_ocr_line_not_duplicated_when_cells_represent_it(self):
        raw = fixture()
        raw["pages"][0]["texts"] = [{"id": "merged", "value": "Item 1 1.00", "original_value": "Item 1 1.00", "score": 0.9, "bbox": {"x_min": 5, "y_min": 50, "x_max": 160, "y_max": 65}}]
        page = present_document(raw)["pages"][0]

        self.assertEqual(len(page["regions"]), 2)

    def test_duplicate_region_page_and_mixed_formats_rejected(self):
        url = self.seed(fixture())
        region = {"id": "text:title", "value": "bad"}
        payloads = [{"pages": [{"page_index": 0, "regions": [region, region]}]}, {"pages": [{"page_index": 0, "regions": [region]}] * 2}, {"regions": [region], "texts": [{"id": "title", "value": "other"}]}]

        for payload in payloads:
            self.assertEqual(self.client.patch(url, json=payload).status_code, 422)

        self.assertEqual(next(r for r in self.client.get(url).json()["pages"][0]["regions"] if r["id"] == "text:title")["value"], "Packing List")

    def test_overlapping_boxes_do_not_double_coverage(self):
        box = {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 20}
        half = {**box, "x_max": 50}

        self.assertEqual(covered_fraction(box, [half, half]), 0.5)
