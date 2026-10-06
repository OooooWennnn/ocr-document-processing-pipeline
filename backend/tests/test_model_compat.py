import sys
import types
import unittest
from unittest.mock import patch
from src.services import paddlex_compat


class ModelCompatTests(unittest.TestCase):
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
