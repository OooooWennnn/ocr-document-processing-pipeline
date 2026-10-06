import unittest
from types import SimpleNamespace
from src.schemas.schema import BBox, Cell, TextItem
from src.services.table_extraction import TableExtraction
from src.services.result_editing import apply_edits, present_table


class SingleTableTests(unittest.TestCase):
    def setUp(self):
        self.extractor = TableExtraction.__new__(TableExtraction)
        self.result = self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 200, 50)), Cell(-1, -1, BBox(0, 50, 100, 100)), Cell(-1, -1, BBox(100, 50, 200, 100))], [TextItem("original", 0.7, BBox(10, 10, 50, 30))])

    def test_merged_cell_and_exact_boundaries(self):
        table = present_table(self.result)

        self.assertEqual(table["cells"][0]["colspan"], 2)
        self.assertEqual(table["x_lines"], self.result["x_lines"])
        self.assertEqual(table["cells"][0]["score"], 0.7)

    def test_edits_round_trip_without_changing_ocr(self):
        edited = apply_edits(self.result, [SimpleNamespace(row=0, col=0, value=""), SimpleNamespace(row=1, col=1, value="수정\n내용")])
        table = present_table(edited)

        self.assertEqual(table["cells"][0]["value"], "")
        self.assertEqual(table["cells"][2]["value"], "수정\n내용")
        self.assertEqual(table["cells"][0]["score"], 0.7)
        self.assertEqual(present_table(self.result)["cells"][0]["value"], "original")

        edited = apply_edits(edited, [SimpleNamespace(row=1, col=0, value="second")])

        self.assertEqual(present_table(edited)["cells"][2]["value"], "수정\n내용")

    def test_invalid_and_duplicate_edits_rejected(self):
        for edits in [[SimpleNamespace(row=9, col=0, value="bad")], [SimpleNamespace(row=0, col=0, value="a")] * 2]:
            with self.assertRaises(ValueError):
                apply_edits(self.result, edits)

    def test_legacy_result_uses_same_boundaries(self):
        legacy = {k: v for k, v in self.result.items() if k not in ("x_lines", "y_lines")}

        self.assertEqual(present_table(legacy), present_table(self.result))

    def test_zero_and_multiple_tables_rejected(self):
        for tables in [[], [{}, {}]]:
            self.extractor.model = SimpleNamespace(predict=lambda image, **kwargs: iter([{"table_res_list": tables}]))

            with self.assertRaises(ValueError):
                self.extractor.validate_table_count({"table_res_list": tables})

    def test_overlapping_cells_rejected(self):
        with self.assertRaises(ValueError):
            self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 100, 100)), Cell(-1, -1, BBox(0, 0, 100, 100))], [])


class MappingAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.extractor = TableExtraction.__new__(TableExtraction)

    def test_table_recognition_reuses_ocr_when_rotation_is_disabled(self):
        recognizer = object()
        table = SimpleNamespace(general_ocr_pipeline=None)
        pipeline = SimpleNamespace(_pipeline=SimpleNamespace(general_ocr_pipeline=recognizer, table_recognition_pipeline=SimpleNamespace(_pipeline=table)))

        TableExtraction.share_table_ocr(pipeline)
        self.assertIs(table.general_ocr_pipeline, recognizer)

        existing = object()
        table.general_ocr_pipeline = existing

        TableExtraction.share_table_ocr(pipeline)
        self.assertIs(table.general_ocr_pipeline, existing)

    def test_narrow_columns_survive_clustering(self):
        result = self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 6, 30)), Cell(-1, -1, BBox(6, 0, 12, 30))], [])

        self.assertEqual(result["x_lines"], [0, 6, 12])
        self.assertEqual([c["col"] for c in result["table_rows"][0]["cells"]], [0, 1])

    def test_ambiguous_text_is_not_silently_put_in_first_cell(self):
        cells = [Cell(-1, -1, BBox(0, 0, 50, 30)), Cell(-1, -1, BBox(50, 0, 100, 30))]
        result = self.extractor.reconstruct_table(cells, [TextItem("crosses fields", 0.9, BBox(40, 5, 60, 20))])

        self.assertEqual(len(result["unmapped_texts"]), 1)
        self.assertTrue(all(c["mapping_warning"] for c in result["table_rows"][0]["cells"]))
        self.assertTrue(all(not c["texts"] for c in result["table_rows"][0]["cells"]))

    def test_overlap_mapping_and_multiline_reading_order(self):
        texts = [TextItem("second line", 0.8, BBox(5, 35, 80, 45)), TextItem("right", 0.9, BBox(45, 10, 80, 20)), TextItem("left", 0.9, BBox(5, 11, 35, 21))]
        table = present_table(self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 100, 60))], texts))

        self.assertEqual(table["cells"][0]["value"], "left right\nsecond line")
        self.assertEqual(table["cells"][0]["mapping_score"], 1)

    def test_split_texts_with_fewer_scores_are_all_preserved(self):
        data = {"overall_ocr_res": {"rec_texts": ["AB"], "rec_boxes": [[0, 0, 100, 20]], "rec_scores": [0.99]}, "table_res_list": [{"table_ocr_pred": {"rec_texts": ["A", "B"], "rec_boxes": [[0, 0, 40, 20], [60, 0, 100, 20]], "rec_scores": [0.99]}}]}
        texts = self.extractor.extract_texts(data)

        self.assertEqual([t.value for t in texts], ["A", "B"])
        self.assertTrue(all(t.score is None for t in texts))

    def test_outside_caption_is_not_assigned_to_table(self):
        result = self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 100, 40))], [TextItem("caption", 0.9, BBox(0, 50, 90, 60))])

        self.assertEqual(result["unmapped_texts"], [])
        self.assertEqual(result["table_rows"][0]["cells"][0]["texts"], [])

    def test_unknown_score_does_not_become_confident(self):
        table = present_table(self.extractor.reconstruct_table([Cell(-1, -1, BBox(0, 0, 100, 40))], [TextItem("one", 0.99, BBox(0, 5, 25, 15)), TextItem("two", None, BBox(30, 5, 50, 15))]))

        self.assertIsNone(table["cells"][0]["score"])


if __name__ == "__main__":
    unittest.main()
