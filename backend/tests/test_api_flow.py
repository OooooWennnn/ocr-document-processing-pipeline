"""API checks with in-memory SQLite and a mocked queue, never the project database."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["REDIS_URL"] = "redis://localhost:63797/15"

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB


@compiles(JSONB, "sqlite")
def jsonb_as_json(type_, compiler, **kw):
    return "JSON"


engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
module = types.ModuleType("src.db.engine")
module.engine = engine
sys.modules["src.db.engine"] = module

from main import app
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session
from src.models.job_model import Job, JobStatus, JobResult
from src.api.v1.endpoints import jobs
from src.services.table_extraction import TableExtraction
from src.schemas.schema import BBox, Cell, TextItem


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
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

        result = Result(table_res_list=[{"cell_box_list": [[0, 0, 200, 80]], "table_ocr_pred": {"rec_texts": ["text"], "rec_scores": [0.9], "rec_boxes": [[10, 10, 100, 40]]}}])
        x = TableExtraction.__new__(TableExtraction)
        x.model = types.SimpleNamespace(predict=lambda image, **kwargs: [result])
        _, table = x.detect(np.zeros((40, 100, 3), dtype=np.uint8))

        self.assertEqual(table["bbox"], {"x_min": 0.0, "y_min": 0.0, "x_max": 100.0, "y_max": 40.0})
        self.assertEqual(table["image_width"], 100)
        self.assertEqual(table["table_rows"][0]["cells"][0]["texts"][0]["bbox"]["x_min"], 5.0)


if __name__ == "__main__":
    unittest.main()
