"""OCR 보정·모델 호환성·메모리 제한과 워커 종료 처리를 검증한다."""

import math
import os
import sys
import types
import unittest
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from billiard.einfo import ExceptionWithTraceback
from billiard.exceptions import WorkerLostError
from celery.worker.request import Request
from sqlmodel import Session

import test_api as api_tests
from src.models import Job, JobStatus
from src.schemas import BBox, Cell, TextItem
from src.services import paddlex_compat
from src.services.ocr_limits import detection_side_limit
from src.services.ocr_refinement import refine_page
from src.services.result_editing import present_document
from src.services.table_extraction import TableExtraction
from celery_app import DocumentRequest, mark_interrupted_job


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


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((60, 120, 3), 255, dtype=np.uint8)
        self.extractor = TableExtraction.__new__(TableExtraction)

    def tables(self, texts):
        return [self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 60, 50)), Cell(-1, -1, BBox(60, 0, 120, 50))], texts)]

    def test_crossing_line_read_in_cell_crops_with_real_scores(self):
        text = TextItem("wrong combined text", 0.9, BBox(5, 10, 115, 25))
        tables = self.tables([text])

        def recognize(crops):
            self.assertEqual(len(crops), 2)

            return [{"rec_text": "left", "rec_score": 0.94}, {"rec_text": "right", "rec_score": 0.88}]

        out, stats = refine_page(self.image, [text], tables, recognize)

        self.assertEqual([t.value for t in out], ["left", "right"])
        self.assertEqual(stats["accepted_lines"], 1)

        cells = tables[0]["table_rows"][0]["cells"]

        self.assertEqual([c["original_value"] for c in cells], ["left", "right"])
        self.assertTrue(all(c["mapping_warning"] for c in cells))
        self.assertEqual(cells[0]["texts"][0]["score"], 0.94)

    def test_failed_or_missing_prediction_preserves_original(self):
        t = TextItem("uncertain", 0.3, BBox(5, 10, 45, 25))
        tables = self.tables([t])
        original = deepcopy(tables)
        out, stats = refine_page(self.image, [t], tables, lambda crops: [{"rec_text": "other", "rec_score": 0.4}])

        self.assertEqual(out, [t])
        self.assertEqual(tables, original)
        self.assertEqual(stats["accepted_lines"], 0)

        with self.assertRaises(ValueError):
            refine_page(self.image, [t], tables, lambda crops: [])

        self.assertEqual(tables, original)

    def test_crop_budget_and_confident_text_skip_recognition(self):
        t = TextItem("clear", 0.99, BBox(5, 10, 45, 25))

        def forbidden(crops):
            self.fail("Recognizer should not run")

        self.assertEqual(refine_page(self.image, [t], self.tables([t]), forbidden)[0], [t])

        with patch.dict("os.environ", {"OCR_REFINE_MAX_CROPS": "0"}):
            t.score = 0.2

            self.assertEqual(refine_page(self.image, [t], self.tables([t]), forbidden)[0], [t])

    def test_rejected_crossing_crops_leave_budget_for_useful_text(self):
        outside = TextItem("partly outside", 0.95, BBox(-50, 10, 120, 25))
        uncertain = TextItem("uncertain", 0.3, BBox(5, 35, 45, 45))
        tables = self.tables([outside, uncertain])

        def recognize(crops):
            self.assertEqual(len(crops), 1)

            return [{"rec_text": "corrected", "rec_score": 0.95}]

        with patch.dict("os.environ", {"OCR_REFINE_MAX_CROPS": "2"}):
            out, stats = refine_page(self.image, [outside, uncertain], tables, recognize)

        self.assertEqual([text.value for text in out], ["partly outside", "corrected"])
        self.assertEqual(stats, {"attempted_crops": 1, "accepted_lines": 1})
        self.assertEqual(tables[0]["table_rows"][0]["cells"][0]["original_value"], "corrected")

    def test_stacked_lines_are_read_separately_within_budget(self):
        for top in [5, 22, 39]:
            self.image[top:top + 8, 12:30] = 0

        text = TextItem("merged", 0.4, BBox(8, 2, 35, 50))

        def recognize(crops):
            self.assertEqual(len(crops), 3)

            return [{"rec_text": value, "rec_score": 0.96} for value in ["10", "20", "30"]]

        neighbor = TextItem("adjacent", 0.98, BBox(40, 5, 58, 13))
        out, stats = refine_page(self.image, [text, neighbor], [], recognize)

        self.assertEqual([item.value for item in out], ["adjacent", "10", "20", "30"])
        self.assertEqual(stats, {"attempted_crops": 3, "accepted_lines": 1})
        self.assertEqual([round((item.bbox.y_min + item.bbox.y_max) / 2) for item in out[1:]], [9, 26, 43])

        with patch.dict("os.environ", {"OCR_REFINE_MAX_CROPS": "2"}):
            out, stats = refine_page(self.image, [text, neighbor], [], lambda crops: self.fail("Crop budget exceeded"))

        self.assertEqual(out, [text, neighbor])
        self.assertEqual(stats["attempted_crops"], 0)

    def test_uncertain_split_fragment_preserves_the_whole_original(self):
        for top in [5, 22, 39]:
            self.image[top:top + 8, 12:30] = 0

        text = TextItem("original", 0.4, BBox(8, 2, 35, 50))
        predictions = [{"rec_text": "10", "rec_score": 0.96}, {"rec_text": "20", "rec_score": 0.6}, {"rec_text": "30", "rec_score": 0.96}]
        out, stats = refine_page(self.image, [text], [], lambda crops: predictions)

        self.assertEqual(out, [text])
        self.assertEqual(stats["accepted_lines"], 0)

    def test_majority_overlap_cannot_assign_whole_string_to_wrong_field(self):
        t = TextItem("two fields", 0.95, BBox(30, 10, 75, 25))
        table = self.tables([t])[0]

        self.assertEqual(len(table["unmapped_texts"]), 1)
        self.assertTrue(all(not c["texts"] for c in table["table_rows"][0]["cells"]))

    def test_fallback_keeps_line_order_confidence_and_token_positions(self):
        tables = self.tables([])
        texts = [{"id": str(i), "value": v, "score": s, "bbox": {"x_min": x, "y_min": 10, "x_max": x + 15, "y_max": 25}} for i, (v, s, x) in enumerate([("left", 0.9, 2), ("right", 0.8, 25)])]
        page = present_document({"schema_version": 3, "page_count": 1, "pages": [{"page_index": 0, "texts": texts, "tables": tables}]})["pages"][0]
        c = next(r for r in page["regions"] if r["id"] == "cell:1:0:0")

        self.assertEqual(c["value"], "left right")
        self.assertEqual(c["score"], 0.8)
        self.assertEqual(c["mapping_score"], 1)
        self.assertEqual([t["bbox"]["x_min"] for t in c["tokens"]], [2, 25])


class ModelCompatTests(unittest.TestCase):
    def test_detector_override_preserves_language_recognizers(self):
        options = []

        class Model:
            def __init__(self, **kwargs):
                options.append(kwargs)
                self.paddlex_pipeline = types.SimpleNamespace()

        module = types.ModuleType("paddleocr")
        module.PPStructureV3 = Model

        with patch.dict(sys.modules, {"paddleocr": module}), patch.object(paddlex_compat, "without_unused_chart", return_value=nullcontext()):
            for language, detector in [("korean", "PP-OCRv5_mobile_det"), ("en", "PP-OCRv5_server_det"), ("fr", "PP-OCRv5_mobile_det")]:
                with patch.dict("os.environ", {"OCR_LANGUAGE": language, "OCR_DETECTION_MODEL": detector}):
                    TableExtraction()

        self.assertEqual(options[0]["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(options[0]["text_recognition_model_name"], "korean_PP-OCRv5_mobile_rec")
        self.assertNotIn("lang", options[0])
        self.assertEqual(options[0]["text_det_unclip_ratio"], 1.0)
        self.assertEqual(options[1]["text_detection_model_name"], "PP-OCRv5_server_det")
        self.assertEqual(options[1]["text_recognition_model_name"], "PP-OCRv5_server_rec")
        self.assertNotIn("lang", options[1])
        self.assertNotIn("text_det_unclip_ratio", options[1])
        self.assertEqual(options[2]["lang"], "fr")
        self.assertNotIn("text_detection_model_name", options[2])

    def test_skip_only_disabled_chart_and_restore_method(self):
        calls = []

        class Pipeline:
            use_chart_recognition = False

            def create_model(self, config, *args, **kwargs):
                calls.append(config)

                return "real-model"

        original = Pipeline.create_model
        module = types.ModuleType("paddlex.inference.pipelines.layout_parsing.pipeline_v2")
        module._LayoutParsingPipelineV2 = Pipeline

        with patch.dict(sys.modules, {module.__name__: module}), patch.object(paddlex_compat, "version", return_value="3.4.1"):
            with paddlex_compat.without_unused_chart():
                obj = Pipeline()

                self.assertIsNone(obj.create_model({"module_name": "chart_recognition"}))
                self.assertEqual(obj.create_model({"module_name": "layout_detection"}), "real-model")

                obj.use_chart_recognition = True

                self.assertEqual(obj.create_model({"module_name": "chart_recognition"}), "real-model")

            self.assertIs(Pipeline.create_model, original)

            with self.assertRaises(ValueError):
                with paddlex_compat.without_unused_chart():
                    raise ValueError("failure")

            self.assertIs(Pipeline.create_model, original)

        self.assertEqual(len(calls), 2)
