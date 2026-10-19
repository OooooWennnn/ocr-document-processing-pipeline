import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch
from src.services import paddlex_compat
from src.services.table_extraction import TableExtraction


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
