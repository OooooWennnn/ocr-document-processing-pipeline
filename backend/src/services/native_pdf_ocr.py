"""Temporarily supply embedded PDF text in place of full-page OCR."""

from contextlib import contextmanager
from unittest.mock import patch


class NativeOCR:
    """Expose native PDF tokens in PaddleX OCRResult format."""
    def __init__(self, original, tokens, scale_x, scale_y):
        """Keep the original OCR object, native tokens and inference scale."""
        self.original, self.tokens = original, tokens
        self.scale_x, self.scale_y = scale_x, scale_y

    def __getattr__(self, name):
        """Delegate other attributes to the original OCR object."""
        return getattr(self.original, name)

    def __call__(self, images, **kwargs):
        """Yield scaled native tokens as OCRResult objects.
        Internal 1.0 scores satisfy PaddleX; API scores remain None."""
        import numpy as np
        from paddlex.inference.pipelines.ocr.result import OCRResult

        for image in images:
            boxes = []

            for token in self.tokens:
                bbox = token["bbox"]
                box = [
                    bbox["x_min"] * self.scale_x,
                    bbox["y_min"] * self.scale_y,
                    bbox["x_max"] * self.scale_x,
                    bbox["y_max"] * self.scale_y,
                ]
                boxes.append(box)

            polys = [np.array([[a, b], [c, b], [c, d], [a, d]], dtype=np.float32) for a, b, c, d in boxes]
            result = {
                "input_path": None,
                "page_index": None,
                "input_img": image,
                "doc_preprocessor_res": {"output_img": image},
                "dt_polys": polys,
                "rec_polys": list(polys),
                "rec_boxes": np.array(boxes, dtype=np.int32),
                "rec_texts": [token["value"] for token in self.tokens],
                "rec_scores": [1.0] * len(boxes),
                "text_type": "general",
                "vis_fonts": [],
                "model_settings": {"use_doc_preprocessor": False, "use_textline_orientation": False},
                "text_det_params": {},
                "text_rec_score_thresh": 0.0,
                "textline_orientation_angles": [-1] * len(boxes),
                "return_word_box": False,
            }

            yield OCRResult(result)


@contextmanager
def use_native_text(model, tokens, scale_x, scale_y):
    """Swap in native OCR during inference and restore the original object on exit."""
    if not tokens:
        yield
        return

    layout = getattr(model.paddlex_pipeline, "_pipeline", model.paddlex_pipeline)
    original = layout.general_ocr_pipeline

    with patch.object(layout, "general_ocr_pipeline", NativeOCR(original, tokens, scale_x, scale_y)):
        yield
