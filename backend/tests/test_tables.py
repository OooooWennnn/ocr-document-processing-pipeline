"""표 경계·빈 셀·병합 셀과 글자 매핑을 검증한다."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.schemas import BBox, Cell, TextItem
from src.services.cell_geometry import recover_ruled_cells, refine_cell_geometry
from src.services.result_editing import apply_edits, present_table
from src.services.table_extraction import TableExtraction


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


class GeometryTests(unittest.TestCase):
    def test_split_only_when_rule_crosses_entire_cell(self):
        image = np.full((100, 130, 3), 255, np.uint8)

        cv2.line(image, (10, 10), (120, 10), (0, 0, 0), 1)
        cv2.line(image, (10, 80), (120, 80), (0, 0, 0), 1)
        cv2.line(image, (10, 45), (120, 45), (0, 0, 0), 1)

        cells = refine_cell_geometry(image, [Cell(-1, -1, BBox(10, 10, 120, 80))])

        self.assertEqual([(c.bbox.y_min, c.bbox.y_max) for c in cells], [(10, 45), (45, 80)])

    def test_partial_line_preserves_merged_field(self):
        image = np.full((100, 130, 3), 255, np.uint8)

        cv2.line(image, (70, 45), (120, 45), (0, 0, 0), 1)
        self.assertEqual(len(refine_cell_geometry(image, [Cell(-1, -1, BBox(10, 10, 120, 80))])), 1)

    def test_wireless_cells_preserve_coordinates_and_order(self):
        image = np.full((100, 130, 3), 255, np.uint8)
        box = BBox(10.2, 10.5, 110.3, 80.1)
        cells = refine_cell_geometry(image, [Cell(-1, -1, box)])

        self.assertEqual(cells[0].bbox, box)


class RuledRecoveryTests(unittest.TestCase):
    def make_grid(self):
        """Build a faint grid with a merged header and no text."""
        image = np.full((180, 330, 3), 236, np.uint8)
        cv2.rectangle(image, (20, 20), (300, 60), (170, 190, 175), -1)

        for y in (60, 100, 140):
            cv2.line(image, (20, y), (300, y), (210, 205, 200), 2)

        for x in (20, 300):
            cv2.line(image, (x, 20), (x, 140), (210, 205, 200), 2)

        for x in (100, 260):
            cv2.line(image, (x, 60), (x, 140), (210, 205, 200), 2)

        return image

    def test_empty_faint_grid_recovers_body_and_merged_header(self):
        from src.services.table_extraction import TableExtraction
        from src.services.result_editing import present_table

        image = self.make_grid()
        original = image.copy()
        predicted = [Cell(-1, -1, BBox(20, 20, 300, 140))] * 2
        cells = recover_ruled_cells(image, predicted)
        table = present_table(TableExtraction.__new__(TableExtraction).reconstruct_table(cells, []))

        self.assertEqual(len(table['cells']), 7)
        self.assertEqual(table['cells'][0]['colspan'], 3)
        self.assertEqual([c['row'] for c in table['cells']], [0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(all(c['value'] == '' for c in table['cells']))
        self.assertTrue(all(c['score'] is None for c in table['cells']))
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(predicted[0].bbox, BBox(20, 20, 300, 140))

    def test_open_grid_is_not_filled_with_guessed_cells(self):
        image = self.make_grid()
        image[102:150, 15:26] = 236

        self.assertEqual(recover_ruled_cells(image, [Cell(-1, -1, BBox(20, 20, 300, 140))]), [])

    def test_missing_rules_do_not_create_a_table(self):
        image = np.full((180, 330, 3), 236, np.uint8)
        cv2.putText(image, 'ITEM DESCRIPTION', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        self.assertEqual(recover_ruled_cells(image, [Cell(-1, -1, BBox(20, 20, 300, 140))]), [])
        self.assertEqual(recover_ruled_cells(image, []), [])


class ShallowTableTests(unittest.TestCase):
    def make_order_table(self):
        """Build a shallow table with column rules only in its input row."""
        image = np.full((150, 920, 3), 236, np.uint8)
        cv2.rectangle(image, (20, 20), (880, 60), (170, 190, 175), -1)

        for y in (60, 100):
            cv2.line(image, (20, y), (880, y), (210, 205, 200), 2)

        for x in (20, 880):
            cv2.line(image, (x, 20), (x, 100), (210, 205, 200), 2)

        for x in (300, 580):
            cv2.line(image, (x, 60), (x, 100), (210, 205, 200), 2)

        return image

    def test_short_column_rules_recover_merged_header_and_three_inputs(self):
        from src.services.table_extraction import TableExtraction
        from src.services.result_editing import present_table

        image = self.make_order_table()
        predicted = [Cell(-1, -1, BBox(20, 20, 300, 100)), Cell(-1, -1, BBox(300, 28, 580, 100)), Cell(-1, -1, BBox(600, 24, 650, 45)), Cell(-1, -1, BBox(580, 65, 880, 100))]
        extractor = TableExtraction.__new__(TableExtraction)
        before = present_table(extractor.reconstruct_table(predicted, []))
        cells = refine_cell_geometry(image, predicted)
        after = present_table(extractor.reconstruct_table(cells, []))

        self.assertGreater(len(before["y_lines"]), 3)
        self.assertEqual(len(after["y_lines"]), 3)
        self.assertEqual(len(after["x_lines"]), 4)
        self.assertEqual(len(after["cells"]), 4)
        self.assertEqual(after["cells"][0]["colspan"], 3)
        self.assertEqual([(c["row"], c["col"]) for c in after["cells"]], [(0, 0), (1, 0), (1, 1), (1, 2)])

    def test_partial_grid_keeps_additional_model_cells(self):
        image = self.make_order_table()
        predicted = [Cell(-1, -1, BBox(20, 20, 300, 60)), Cell(-1, -1, BBox(300, 20, 580, 60)), Cell(-1, -1, BBox(580, 20, 880, 60)), Cell(-1, -1, BBox(20, 60, 300, 100)), Cell(-1, -1, BBox(300, 60, 580, 100)), Cell(-1, -1, BBox(580, 60, 880, 100))]

        self.assertEqual(len(recover_ruled_cells(image, predicted)), 4)
        self.assertEqual(len(refine_cell_geometry(image, predicted)), 6)
