import io
import types
from unittest.mock import patch
import numpy as np
import unittest
import test_api_flow as api_tests
from test_api_flow import engine
from sqlmodel import Session
from src.models.job_model import Job, JobResult, JobStatus
from src.api.v1.endpoints import jobs
from src.services.document_input import validate_pdf, iter_pages
from src.services.table_extraction import TableExtraction
from src.tasks import ocr_tasks


def pdf_bytes():
    stream = b"BT /F1 18 Tf 30 130 Td (Packing List) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 160] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 160] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    output = b"%PDF-1.4\n"
    offsets = []

    for i, obj in enumerate(objects, 1):
        offsets.append(len(output))

        output += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    start = len(output)
    output += b"xref\n0 7\n0000000000 65535 f \n" + b"".join(f"{n:010d} 00000 n \n".encode() for n in offsets)

    return output + f"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode()


class DocumentFlowTests(unittest.TestCase):
    setUp = api_tests.ApiFlowTests.setUp
    tearDown = api_tests.ApiFlowTests.tearDown

    def test_scoped_edits(self):
        from src.schemas.schema import Cell, BBox, TextItem

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

        with patch.object(ocr_tasks, "get_extractor", return_value=types.SimpleNamespace(detect_document=detect)):
            ocr_tasks.process_job.run(job_id)

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
        from test_cell_geometry import RuledRecoveryTests
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
        from test_cell_geometry import ShallowTableTests
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
