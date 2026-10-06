import math
import os
import types
import unittest
from unittest.mock import patch
import numpy as np
import test_api_flow as api_tests
from sqlmodel import Session
from billiard.einfo import ExceptionWithTraceback
from billiard.exceptions import WorkerLostError
from celery.worker.request import Request
from src.models.job_model import Job, JobStatus
from src.services.ocr_limits import detection_side_limit
from src.services.table_extraction import TableExtraction
from src.tasks.ocr_task_base import DocumentRequest, mark_interrupted_job


class DetectorBudgetTests(unittest.TestCase):
    def test_rounded_detector_inputs_fit_budget_for_page_shapes(self):
        with patch.dict(os.environ, {"OCR_DETECTION_MAX_PIXELS": "1000000", "OCR_DETECTION_SIZE": "2048"}):
            for height, width in [(1126, 1600), (1600, 1600), (4000, 3000), (400, 8000)]:
                limit = detection_side_limit((height, width, 3))
                longest, shortest = max(height, width), min(height, width)

                self.assertLessEqual(limit * math.ceil(shortest * limit / longest / 32) * 32, 1000000)
                self.assertLessEqual(limit, 2048)

    def test_detector_budget_does_not_resize_recognition_source(self):
        class Result(dict):
            @property
            def json(self):
                return dict(self)

        calls = []

        def predict(image, **kwargs):
            calls.append((image.shape, kwargs))

            return [Result(overall_ocr_res={"rec_texts": ["title"], "rec_scores": [0.9], "rec_boxes": [[10, 10, 60, 30]]}, table_res_list=[])]

        extractor = TableExtraction.__new__(TableExtraction)
        extractor.model = types.SimpleNamespace(predict=predict)

        with patch.dict(os.environ, {"OCR_UPSCALE_TARGET": "1600", "OCR_DETECTION_MAX_PIXELS": "1000000"}):
            _, page = extractor.detect_document(np.zeros((563, 800, 3), np.uint8))

        self.assertEqual(calls[0][0], (1126, 1600, 3))
        self.assertLess(calls[0][1]["text_det_limit_side_len"], 1600)
        self.assertEqual(page["texts"][0]["bbox"]["x_min"], 5)


class WorkerFailureTests(unittest.TestCase):
    setUp = api_tests.ApiFlowTests.setUp
    tearDown = api_tests.ApiFlowTests.tearDown

    def create_job(self, status=JobStatus.processing):
        with Session(api_tests.engine) as session:
            session.add(Job(id=81, original_filename="test.png", file_type="image/png", status=status))
            session.commit()

    def request(self):
        request = DocumentRequest.__new__(DocumentRequest)
        request._args = (81,)
        request._kwargs = {}

        return request

    def test_child_sigkill_is_persisted_as_failure_by_parent(self):
        self.create_job()

        exc_info = types.SimpleNamespace(exception=ExceptionWithTraceback(WorkerLostError("SIGKILL"), None))

        with patch.object(Request, "on_failure"):
            self.request().on_failure(exc_info)

        with Session(api_tests.engine) as session:
            self.assertEqual(session.get(Job, 81).status, JobStatus.failed)
            self.assertIn("stopped unexpectedly", session.get(Job, 81).error_message)

    def test_hard_timeout_marks_failure_and_soft_timeout_does_not(self):
        self.create_job()

        with patch.object(Request, "on_timeout"):
            self.request().on_timeout(True, 1800)

            with Session(api_tests.engine) as session:
                self.assertEqual(session.get(Job, 81).status, JobStatus.processing)

            self.request().on_timeout(False, 1860)

        with Session(api_tests.engine) as session:
            self.assertEqual(session.get(Job, 81).status, JobStatus.failed)

    def test_completed_results_are_not_overwritten_by_late_failure(self):
        self.create_job(JobStatus.done)
        mark_interrupted_job(81, "unexpected stop")

        with Session(api_tests.engine) as session:
            self.assertEqual(session.get(Job, 81).status, JobStatus.done)
