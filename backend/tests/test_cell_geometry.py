import unittest
import numpy as np
import cv2
from src.schemas.schema import BBox, Cell
from src.services.cell_geometry import recover_ruled_cells, refine_cell_geometry


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
