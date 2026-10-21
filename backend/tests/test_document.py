"""문서 전체의 추출 결과, 여러 페이지 저장과 영역 매핑을 검증한다."""

import io
import types
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from sqlmodel import Session

import test_api as api_tests
from test_api import engine, pdf_bytes
from src import api as jobs
from src.models import Job, JobResult, JobStatus
from src.schemas import BBox, Cell, TextItem
from src.services import document_job
from src.services.document_input import iter_pages, validate_pdf
from src.services.document_layout import covered_fraction
from src.services.result_editing import present_document
from src.services.table_extraction import TableExtraction
import celery_app as worker


class DocumentFlowTests(unittest.TestCase):
    setUp = api_tests.ApiFlowTests.setUp
    tearDown = api_tests.ApiFlowTests.tearDown

    def test_scoped_edits(self):
        from src.schemas import Cell, BBox, TextItem

        x = TableExtraction.__new__(TableExtraction)
        table = x.reconstruct_table([Cell(-1, -1, BBox(0, 30, 100, 60))], [TextItem("cell", 0.9, BBox(5, 35, 30, 50))])
        document = {"schema_version": 3, "page_count": 2, "pages": [{"page_index": i, "image_width": 100, "image_height": 100, "texts": [{"id": "0", "value": "title", "original_value": "title", "score": 0.9, "bbox": {"x_min": 0, "y_min": 0, "x_max": 50, "y_max": 20}}], "tables": [{**table, "id": "1"}, {**table, "id": "2"}]} for i in range(2)]}

        with Session(engine) as session:
            session.add(Job(id=10, original_filename="two.pdf", file_type="application/pdf", page_count=2, status=JobStatus.done))
            session.commit()
            session.add(JobResult(job_id=10, result_json=document))
            session.commit()

        url = "/api/v1/jobs/10/result"
        response = self.client.patch(url, json={"page_index": 1, "table_id": "2", "cells": [{"row": 0, "col": 0, "value": "second table"}], "texts": [{"id": "0", "value": "second page title"}]})

        self.assertEqual(response.status_code, 200, response.text)

        pages = self.client.get(url).json()["pages"]

        self.assertEqual(pages[0]["texts"][0]["value"], "title")
        self.assertEqual(pages[1]["texts"][0]["value"], "second page title")
        self.assertEqual(pages[1]["texts"][0]["original_value"], "title")
        self.assertEqual(pages[1]["tables"][0]["cells"][0]["value"], "cell")
        self.assertEqual(pages[1]["tables"][1]["cells"][0]["value"], "second table")

        for payload in [{"page_index": 9, "texts": [{"id": "0", "value": "bad"}]}, {"page_index": 1, "texts": [{"id": "missing", "value": "bad"}]}, {"page_index": 1, "table_id": "missing", "cells": [{"row": 0, "col": 0, "value": "bad"}]}, {}]:
            self.assertEqual(self.client.patch(url, json=payload).status_code, 422)

    def test_pdf_upload_worker_and_preview(self):
        content = pdf_bytes()

        self.assertEqual(validate_pdf(content), 2)

        with patch.object(jobs.process_job, "delay"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("two.pdf", content, "application/pdf")})

        self.assertEqual(response.status_code, 201, response.text)

        job_id = response.json()["job_id"]

        with Session(engine) as session:
            job = session.get(Job, job_id)
            source = job.file_path

            self.assertEqual(job.page_count, 2)

        rendered = list(iter_pages(source, "application/pdf"))

        self.assertEqual(len(rendered), 2)
        self.assertEqual(rendered[0][2][0]["value"], "Packing List")

        calls = []

        def detect(image, **kwargs):
            if calls:
                progress = self.client.get(f"/api/v1/jobs/{job_id}/result").json()

                self.assertEqual(progress["status"], "processing")
                self.assertEqual(progress["completed_pages"], 1)

            calls.append(1)

            return {}, {"image_width": image.shape[1], "image_height": image.shape[0], "texts": [], "tables": [], "warnings": [], "elapsed_seconds": 0.01}

        with patch.object(document_job, "get_extractor", return_value=types.SimpleNamespace(detect_document=detect)):
            worker.process_job.run(job_id)

        result = self.client.get(f"/api/v1/jobs/{job_id}/result").json()

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["completed_pages"], 2)
        self.assertEqual(result["pages"][1]["texts"][0]["value"], "Packing List")
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/image?page_index=1").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/image?page_index=2").status_code, 404)

    def test_text_only_multiple_and_invalid_tables(self):
        class Result(dict):
            @property
            def json(self):
                return dict(self)

        x = TableExtraction.__new__(TableExtraction)
        overall = {"rec_texts": ["Packing List", "Notes"], "rec_scores": [0.99, 0.95], "rec_boxes": [[10, 10, 120, 30], [10, 150, 100, 170]]}

        for tables, expected in [([], 0), ([{"cell_box_list": [[0, 40, 120, 80]]}, {"cell_box_list": [[0, 90, 120, 130]]}], 2), ([{"cell_box_list": [[0, 20, 100, 60], [0, 20, 100, 60]]}], 0)]:
            x.model = types.SimpleNamespace(predict=lambda image, **kwargs: [Result(overall_ocr_res=overall, table_res_list=tables)])
            _, page = x.detect_document(np.zeros((200, 200, 3), dtype=np.uint8))

            self.assertEqual([t["value"] for t in page["texts"]], ["Packing List", "Notes"])
            self.assertEqual(len(page["tables"]), expected)

            if tables and not expected:
                self.assertTrue(page["warnings"])

    def test_overlapping_model_cells_recover_without_losing_text_or_empty_cells(self):
        from test_tables import RuledRecoveryTests
        from src.services.result_editing import present_document

        class Result(dict):
            @property
            def json(self):
                return dict(self)

        image = RuledRecoveryTests().make_grid()
        overall = {"rec_texts": ["ITEM"], "rec_scores": [0.98], "rec_boxes": [[60, 60, 140, 100]]}
        result = Result(overall_ocr_res=overall, table_res_list=[{"cell_box_list": [[40, 40, 600, 280]] * 2}])
        extractor = TableExtraction.__new__(TableExtraction)
        extractor.model = types.SimpleNamespace(predict=lambda image, **kwargs: [result])
        _, page = extractor.detect_document(image)
        page['page_index'] = 0
        document = present_document({"schema_version": 3, "page_count": 1, "pages": [page]})
        visible = document['pages'][0]

        self.assertEqual(len(visible['tables']), 1)
        self.assertEqual(len(visible['regions']), 7)
        self.assertEqual(visible['tables'][0]['cells'][0]['value'], 'ITEM')
        self.assertEqual(visible['tables'][0]['cells'][0]['score'], 0.98)
        self.assertEqual(sum(c['value'] == '' for c in visible['regions']), 6)
        self.assertEqual(visible['texts'], [])
        self.assertFalse(any('overlap' in warning for warning in visible['warnings']))

    def test_shallow_order_table_aligns_headers_and_preserves_empty_inputs(self):
        from test_tables import ShallowTableTests
        from src.services.result_editing import present_document

        class Result(dict):
            @property
            def json(self):
                return dict(self)

        image = ShallowTableTests().make_order_table()
        # The 150x920 input is upscaled 2x, so mocked coordinates must match.
        boxes = [[20, 20, 300, 100], [300, 28, 580, 100], [600, 24, 650, 45], [580, 65, 880, 100]]
        overall = {"rec_texts": ["ORDER DATE", "ORDER NUMBER", "JOB"], "rec_scores": [0.98, 0.97, 0.96], "rec_boxes": [[60, 60, 260, 90], [700, 60, 960, 90], [1260, 60, 1340, 90]]}
        result = Result(overall_ocr_res=overall, table_res_list=[{"cell_box_list": [[v * 2 for v in box] for box in boxes]}])
        extractor = TableExtraction.__new__(TableExtraction)
        extractor.model = types.SimpleNamespace(predict=lambda image, **kwargs: [result])
        _, page = extractor.detect_document(image)
        page["page_index"] = 0
        document = present_document({"schema_version": 3, "page_count": 1, "pages": [page]})
        visible = document["pages"][0]
        table = visible["tables"][0]
        header = visible["regions"][0]

        self.assertEqual(len(table["y_lines"]), 3)
        self.assertEqual(len(table["x_lines"]), 4)
        self.assertEqual(len(table["cells"]), 4)
        self.assertEqual([t["value"] for t in header["tokens"]], ["ORDER DATE", "ORDER NUMBER", "JOB"])
        self.assertEqual(header["colspan"], 3)
        self.assertEqual(header["score"], 0.96)
        self.assertTrue(all(c["value"] == "" for c in table["cells"][1:]))
        self.assertEqual(visible["texts"], [])

    def test_corrupt_and_excessive_pdf(self):
        self.assertEqual(self.client.post("/api/v1/jobs/", files={"file": ("broken.pdf", b"%PDF-invalid", "application/pdf")}).status_code, 422)
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument.new()

        for i in range(31):
            pdf.new_page(100, 100).close()

        buffer = io.BytesIO()

        pdf.save(buffer)
        pdf.close()
        self.assertEqual(self.client.post("/api/v1/jobs/", files={"file": ("many.pdf", buffer.getvalue(), "application/pdf")}).status_code, 422)


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
