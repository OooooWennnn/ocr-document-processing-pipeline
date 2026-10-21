"""API checks with in-memory SQLite and a mocked queue, never the project database."""

import io
import os
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, select

os.environ["REDIS_URL"] = "redis://localhost:63797/15"


@compiles(JSONB, "sqlite")
def jsonb_as_json(type_, compiler, **kw):
    return "JSON"


engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

# 앱이 시작되기 전에 테스트 전용 DB 연결로 교체한다.
with patch.dict(os.environ, {"DATABASE_URL": "sqlite://"}):
    from src import database

database.engine = engine

from main import app
from src import api as jobs
from src.models import Job, JobResult, JobStatus
from src.schemas import BBox, Cell, TextItem
from src.services import document_job, document_storage
from src.services.table_extraction import TableExtraction
import celery_app as worker


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


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
        # 실제 .env에 버킷이 설정돼 있어도 일반 API 테스트는 로컬 임시 저장소만 사용한다.
        self.local_storage = patch.dict(os.environ, {"OCR_S3_BUCKET": ""})
        self.local_storage.start()
        self.addCleanup(self.local_storage.stop)

        SQLModel.metadata.drop_all(engine)
        SQLModel.metadata.create_all(engine)

        self.temp = tempfile.TemporaryDirectory()
        self.storage = patch.object(jobs, "STORAGE_ROOT", Path(self.temp.name))

        self.storage.start()

        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.storage.stop()
        self.temp.cleanup()

    def test_edit_uses_job_id_preserves_original_and_roundtrips(self):
        x = TableExtraction.__new__(TableExtraction)
        original = x.reconstruct_table([Cell(-1, -1, BBox(0, 0, 100, 40))], [TextItem("original", 0.9, BBox(5, 5, 80, 20))])

        with Session(engine) as session:
            session.add(Job(id=42, original_filename="test.png", file_type="image/png", status=JobStatus.done))
            session.commit()
            session.add(JobResult(id=7, job_id=42, result_json=original))
            session.commit()

        response = self.client.patch("/api/v1/jobs/42/result", json={"cells": [{"row": 0, "col": 0, "value": "updated\ntext"}]})

        self.assertEqual(response.status_code, 200, response.text)

        cell = self.client.get("/api/v1/jobs/42/result").json()["tables"][0]["cells"][0]

        self.assertEqual(cell["value"], "updated\ntext")
        self.assertEqual(cell["original_value"], "original")
        self.assertEqual(cell["score"], 0.9)
        self.assertEqual(cell["mapping_score"], 1)
        self.assertEqual(self.client.patch("/api/v1/jobs/42/result", json={"cells": [{"row": 99, "col": 0, "value": "bad"}]}).status_code, 422)

        with Session(engine) as session:
            self.assertEqual(session.get(JobResult, 7).result_json, original)

    def test_latest_result_preserves_edit_priority_without_loading_raw_json(self):
        original = {"schema_version": 3, "page_count": 1, "pages": [{"page_index": 0, "image_width": 100, "image_height": 100, "tables": [], "texts": [{"id": "0", "value": "original", "score": 0.9, "bbox": {"x_min": 0, "y_min": 0, "x_max": 80, "y_max": 20}}]}]}
        edited = deepcopy(original)
        edited["pages"][0]["texts"][0]["value"] = "edited"

        with Session(engine) as session:
            session.add(Job(id=43, original_filename="test.png", file_type="image/png", status=JobStatus.done))
            session.commit()
            session.add(JobResult(id=8, job_id=43, result_json={"schema_version": 3, "pages": []}))
            session.add(JobResult(id=9, job_id=43, raw_json={"unused": "model output" * 1000}, result_json=original))
            session.commit()

        selected_columns = []

        def record_columns(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT") and "jobresult" in statement.lower():
                selected_columns.append([column[0] for column in cursor.description])

        event.listen(engine, "after_cursor_execute", record_columns)

        try:
            for saved_edit in (None, edited):
                with self.subTest(edited=saved_edit):
                    with Session(engine) as session:
                        saved = session.get(JobResult, 9)
                        saved.edited_json = saved_edit
                        session.add(saved)
                        session.commit()

                    selected_columns.clear()
                    response = self.client.get("/api/v1/jobs/43/result")
                    expected = jobs.present_document(original if saved_edit is None else saved_edit)

                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json(), {**expected, "status": "done", "message": None, "page_count": 1})
                    self.assertEqual(selected_columns, [["result_json", "edited_json"]])
        finally:
            event.remove(engine, "after_cursor_execute", record_columns)

    def test_result_without_completed_pages_preserves_status_response(self):
        with Session(engine) as session:
            session.add(Job(id=44, original_filename="pending.pdf", file_type="application/pdf", page_count=3, status=JobStatus.processing))
            session.commit()

        expected = {"status": "processing", "message": None, "page_count": 3}

        self.assertEqual(self.client.get("/api/v1/jobs/44/result").json(), expected)

        with Session(engine) as session:
            session.add(JobResult(job_id=44, result_json={}, edited_json={"schema_version": 3, "pages": []}))
            session.commit()

        self.assertEqual(self.client.get("/api/v1/jobs/44/result").json(), expected)

    def test_image_upload_and_preview_have_identical_normalized_pixels(self):
        import io
        from PIL import Image

        content = io.BytesIO()

        Image.new("RGBA", (30, 20), (0, 0, 0, 0)).save(content, format="PNG")

        with patch.object(jobs.process_job, "delay"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("../../unsafe.png", content.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 201, response.text)

        job_id = response.json()["job_id"]
        image = self.client.get(f"/api/v1/jobs/{job_id}/image")

        self.assertEqual(image.status_code, 200)

        normalized = Image.open(io.BytesIO(image.content))

        self.assertEqual(normalized.getpixel((0, 0)), (255, 255, 255))

        with Session(engine) as session:
            self.assertTrue(Path(session.get(Job, job_id).file_path).is_relative_to(self.temp.name))

    def test_queue_failure_has_english_error_and_invalid_upload_is_rejected(self):
        import io
        from PIL import Image

        data = io.BytesIO()

        Image.new("RGB", (20, 20)).save(data, format="PNG")

        with patch.object(jobs.process_job, "delay", side_effect=RuntimeError("offline")):
            response = self.client.post("/api/v1/jobs/", files={"file": ("test.png", data.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 503)
        self.assertIn("queue", response.json()["detail"])
        self.assertEqual(self.client.post("/api/v1/jobs/", files={"file": ("bad.png", b"invalid", "image/png")}).status_code, 415)
        self.assertEqual(self.client.get("/api/v1/jobs/999/image").status_code, 404)

    def test_detect_rescales_coordinates_to_original_preview(self):
        import numpy as np

        class Result(dict):
            @property
            def json(self):
                return dict(self)

        result = Result(table_res_list=[{"cell_box_list": [[0, 0, 180, 60]], "table_ocr_pred": {"rec_texts": ["text"], "rec_scores": [0.9], "rec_boxes": [[10, 10, 100, 40]]}}])
        x = TableExtraction.__new__(TableExtraction)
        x.model = types.SimpleNamespace(predict=lambda image, **kwargs: [result])
        _, page = x.detect_document(np.full((40, 100, 3), 255, dtype=np.uint8))
        table = page["tables"][0]

        self.assertEqual(table["bbox"], {"x_min": 0.0, "y_min": 0.0, "x_max": 90.0, "y_max": 30.0})
        self.assertEqual(table["image_width"], 100)
        self.assertEqual(table["table_rows"][0]["cells"][0]["texts"][0]["bbox"]["x_min"], 5.0)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.downloaded_paths = []
        self.bodies = []

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.objects[bucket, key] = (Path(filename).read_bytes(), ExtraArgs["ContentType"])

    def download_file(self, bucket, key, filename):
        self.downloaded_paths.append(Path(filename))
        Path(filename).write_bytes(self.objects[bucket, key][0])

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject")

        content, content_type = self.objects[Bucket, Key]
        stream = io.BytesIO(content)
        self.bodies.append(stream)

        return {"Body": StreamingBody(stream, len(content)), "ContentLength": len(content), "ContentType": content_type}


class S3StorageTests(unittest.TestCase):
    def setUp(self):
        ApiFlowTests.setUp(self)

        self.s3 = FakeS3()
        self.environment = patch.dict(os.environ, {"OCR_S3_BUCKET": "test-documents", "AWS_REGION": "ca-central-1"})
        self.client_patch = patch.object(document_storage, "get_s3_client", return_value=self.s3)
        self.environment.start()
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addCleanup(self.environment.stop)

    tearDown = ApiFlowTests.tearDown

    def upload(self, filename="scan.png", content=None, content_type="image/png"):
        if content is None:
            stream = io.BytesIO()
            Image.new("RGBA", (30, 20), (0, 0, 0, 0)).save(stream, format="PNG")
            content = stream.getvalue()

        with patch.object(jobs.process_job, "delay"):
            response = self.client.post("/api/v1/jobs/", files={"file": (filename, content, content_type)})

        self.assertEqual(response.status_code, 201, response.text)

        return response.json()["job_id"]

    def process(self, job_id):
        def detect(image, **kwargs):
            return {}, {"image_width": image.shape[1], "image_height": image.shape[0], "texts": [], "tables": [], "warnings": [], "elapsed_seconds": 0.01}

        with patch.object(document_job, "get_extractor", return_value=types.SimpleNamespace(detect_document=detect)):
            worker.process_job.run(job_id)

    def test_image_roundtrip_without_shared_api_and_worker_files(self):
        job_id = self.upload(filename="../../unsafe.png")

        with Session(engine) as session:
            self.assertTrue(session.get(Job, job_id).file_path.startswith("s3://test-documents/uploads/"))

        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])
        self.process(job_id)

        response = self.client.get(f"/api/v1/jobs/{job_id}/image")
        image = Image.open(io.BytesIO(response.content))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertTrue(all(stream.closed for stream in self.s3.bodies))
        self.assertTrue(self.s3.downloaded_paths)
        self.assertTrue(all(not path.exists() for path in self.s3.downloaded_paths))
        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/result").json()["status"], "done")

    def test_multipage_pdf_previews_survive_worker_temporary_file_cleanup(self):
        job_id = self.upload("two.pdf", pdf_bytes(), "application/pdf")

        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/image").status_code, 404)
        self.process(job_id)

        result = self.client.get(f"/api/v1/jobs/{job_id}/result").json()

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["completed_pages"], 2)
        self.assertEqual(result["pages"][1]["texts"][0]["value"], "Packing List")

        for page_index in (0, 1):
            response = self.client.get(f"/api/v1/jobs/{job_id}/image?page_index={page_index}")

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(Image.open(io.BytesIO(response.content)).format, "PNG")

        self.assertEqual(self.client.get(f"/api/v1/jobs/{job_id}/image?page_index=2").status_code, 404)
        self.assertEqual(len(self.s3.objects), 3)
        self.assertTrue(all(not path.parent.exists() for path in self.s3.downloaded_paths))

    def test_failed_database_save_removes_uploaded_object_and_local_copy(self):
        stream = io.BytesIO()
        Image.new("RGB", (20, 20)).save(stream, format="PNG")

        with patch.object(Session, "commit", side_effect=RuntimeError("database unavailable")), patch.object(jobs.process_job, "delay") as queue, self.assertLogs(jobs.logger, level="ERROR"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("scan.png", stream.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.s3.objects, {})
        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])
        queue.assert_not_called()

    def test_failed_s3_upload_does_not_create_job_or_queue_task(self):
        stream = io.BytesIO()
        Image.new("RGB", (20, 20)).save(stream, format="PNG")

        with patch.object(self.s3, "upload_file", side_effect=RuntimeError("storage unavailable")), patch.object(jobs.process_job, "delay") as queue, self.assertLogs(jobs.logger, level="ERROR"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("scan.png", stream.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])

        with Session(engine) as session:
            self.assertEqual(session.exec(select(Job)).all(), [])

        queue.assert_not_called()

    def test_s3_access_failure_is_not_reported_as_missing_document(self):
        job_id = self.upload()
        error = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")

        with patch.object(self.s3, "get_object", side_effect=error), self.assertLogs(jobs.logger, level="ERROR"):
            response = self.client.get(f"/api/v1/jobs/{job_id}/image")

        self.assertEqual(response.status_code, 503)
        self.assertIn("storage", response.json()["detail"])
        self.assertNotIn("AccessDenied", response.text)

    def test_download_context_cleans_up_after_processing_error(self):
        self.s3.objects["test-documents", "uploads/source.pdf"] = (b"example", "application/pdf")

        with self.assertRaisesRegex(RuntimeError, "processing failed"):
            with document_storage.local_source("s3://test-documents/uploads/source.pdf") as source:
                self.assertTrue(source.is_file())
                raise RuntimeError("processing failed")

        self.assertFalse(source.parent.exists())
