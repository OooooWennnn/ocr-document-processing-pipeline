import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from src.schemas.schema import BBox, Cell, TextItem
from src.services.table_extraction import TableExtraction
from src.services.ocr_refinement import refine_page
from src.services.result_editing import present_document


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


if __name__ == "__main__":
    unittest.main()
