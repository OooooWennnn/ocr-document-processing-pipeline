"""AWS 연결 없이 원본·미리보기의 서버 간 전달과 실패 시 정리를 검증한다."""

import io
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from PIL import Image
from sqlmodel import Session, select

import test_api_flow as api_tests
from test_document_flow import pdf_bytes
from src.models.job_model import Job, JobStatus
from src.services import document_storage
from src.tasks import ocr_tasks


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
        api_tests.ApiFlowTests.setUp(self)

        self.s3 = FakeS3()
        self.environment = patch.dict(os.environ, {"OCR_S3_BUCKET": "test-documents", "AWS_REGION": "ca-central-1"})
        self.client_patch = patch.object(document_storage, "get_s3_client", return_value=self.s3)
        self.environment.start()
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addCleanup(self.environment.stop)

    tearDown = api_tests.ApiFlowTests.tearDown

    def upload(self, filename="scan.png", content=None, content_type="image/png"):
        if content is None:
            stream = io.BytesIO()
            Image.new("RGBA", (30, 20), (0, 0, 0, 0)).save(stream, format="PNG")
            content = stream.getvalue()

        with patch.object(api_tests.jobs.process_job, "delay"):
            response = self.client.post("/api/v1/jobs/", files={"file": (filename, content, content_type)})

        self.assertEqual(response.status_code, 201, response.text)

        return response.json()["job_id"]

    def process(self, job_id):
        def detect(image, **kwargs):
            return {}, {"image_width": image.shape[1], "image_height": image.shape[0], "texts": [], "tables": [], "warnings": [], "elapsed_seconds": 0.01}

        with patch.object(ocr_tasks, "get_extractor", return_value=types.SimpleNamespace(detect_document=detect)):
            ocr_tasks.process_job.run(job_id)

    def test_image_roundtrip_without_shared_api_and_worker_files(self):
        job_id = self.upload(filename="../../unsafe.png")

        with Session(api_tests.engine) as session:
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

        with patch.object(Session, "commit", side_effect=RuntimeError("database unavailable")), patch.object(api_tests.jobs.process_job, "delay") as queue, self.assertLogs(api_tests.jobs.logger, level="ERROR"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("scan.png", stream.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.s3.objects, {})
        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])
        queue.assert_not_called()

    def test_failed_s3_upload_does_not_create_job_or_queue_task(self):
        stream = io.BytesIO()
        Image.new("RGB", (20, 20)).save(stream, format="PNG")

        with patch.object(self.s3, "upload_file", side_effect=RuntimeError("storage unavailable")), patch.object(api_tests.jobs.process_job, "delay") as queue, self.assertLogs(api_tests.jobs.logger, level="ERROR"):
            response = self.client.post("/api/v1/jobs/", files={"file": ("scan.png", stream.getvalue(), "image/png")})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(list(Path(self.temp.name).rglob("*.png")), [])

        with Session(api_tests.engine) as session:
            self.assertEqual(session.exec(select(Job)).all(), [])

        queue.assert_not_called()

    def test_s3_access_failure_is_not_reported_as_missing_document(self):
        job_id = self.upload()
        error = ClientError({"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")

        with patch.object(self.s3, "get_object", side_effect=error), self.assertLogs(api_tests.jobs.logger, level="ERROR"):
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
