import os
import types
import unittest

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np
from src.services.native_pdf_ocr import use_native_text


class NativePdfTests(unittest.TestCase):
    def test_embedded_text_skips_recognition_serializes_and_restores_pipeline(self):
        original = types.SimpleNamespace(text_rec_model=object())
        layout = types.SimpleNamespace(general_ocr_pipeline=original)
        model = types.SimpleNamespace(paddlex_pipeline=types.SimpleNamespace(_pipeline=layout))
        tokens = [{"value": "Packing List", "bbox": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 40}}]

        with use_native_text(model, tokens, 2, 2):
            proxy = layout.general_ocr_pipeline

            self.assertIs(proxy.text_rec_model, original.text_rec_model)

            result = list(proxy([np.zeros((100, 300, 3), dtype=np.uint8)]))[0]

            self.assertEqual(result["rec_texts"], ["Packing List"])
            self.assertEqual(result["rec_boxes"].tolist(), [[20, 40, 220, 80]])
            self.assertEqual(result.json["res"]["rec_texts"], ["Packing List"])

        self.assertIs(layout.general_ocr_pipeline, original)

        with self.assertRaises(ValueError):
            with use_native_text(model, tokens, 1, 1):
                raise ValueError("fail")

        self.assertIs(layout.general_ocr_pipeline, original)
