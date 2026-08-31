from typing import Optional, Union
import numpy as np
from dataclasses import asdict

from src.schemas.schema import BBox, TextItem, Cell, RawTable, TableRow, Table

class TableExtraction:
    """
    A class for performing table structure and OCR on images using PaddleOCR
    """

    def __init__(self) -> None:
        """
        Initialize the model
        """
        from paddleocr import PPStructureV3
        self.model = PPStructureV3(
            use_doc_unwarping=False, 
            lang='korean',)
        

    def detect(self, image: np.ndarray) -> Optional[tuple]:
        """
        Detect tables and texts in the image

        Args:
            image_path: Path to the input image

        Returns:

        """
        raw_result = self.model.predict(image)

        if not raw_result:
            return None, None
        
        cells = self.extract_table_structure(raw_result[0])
        texts = self.extract_texts(raw_result[0])
        result = self.reconstruct_table(cells, texts)

        return {}, result
    
    def extract_texts(self, data: dict) -> Optional[list[TextItem]]:
        if not data:
            return None
        
        ocr = data['overall_ocr_res']
        texts = ocr['rec_texts']
        scores = ocr['rec_scores']
        bounding_boxes = [box.tolist() for box in ocr['rec_boxes']]
        items: list[TextItem] = []

        for t, s, b in zip(texts, scores, bounding_boxes):
            x_min, y_min, x_max, y_max = b
            items.append(TextItem(
                value = t,
                score = s,
                bbox = BBox(x_min, y_min, x_max, y_max)
            ))

        # texts = [asdict(i) for i in items]
        return items
    
    def extract_cells(self, cell_box_list: list) -> list:
        items: list[Cell] = []

        for bbox in cell_box_list:
            cell_bbox = bbox.tolist()  # nparray to list
            x_min, y_min, x_max, y_max = cell_bbox

            items.append(Cell(
                row = -1, # row not decided yet
                col = -1, # col not decided yet
                bbox = BBox(x_min, y_min, x_max, y_max)
            ))

        # cells = [asdict(i) for i in items]
        
        return items
    
    def extract_raw_tables(self, layout_boxes: list):
        raw_tables: list[RawTable] = []
        counter = 0

        for l in layout_boxes:
            if l["label"] == 'table':
                counter += 1
                bbox = [float(x) for x in l['coordinate']]
                x_min, y_min, x_max, y_max = bbox

                raw_tables.append(RawTable(
                    id = f"table_{counter}",
                    bbox = BBox(x_min, y_min, x_max, y_max)
                ))


    
    def extract_table_structure(self, data: dict) -> Optional[list[Cell]]:
        if not data:
            return None
        
        t_res = data['table_res_list']
        cells: list[Cell] = []

        for i in range(len(t_res)):
            cell_box_list = t_res[i].get("cell_box_list", [])
            cells.extend(self.extract_cells(cell_box_list))
        
        # # table detection
        # layout = data['layout_det_res']['boxes']
        

        return cells

    def reconstruct_table(self, cells: list[Cell], texts: list[TextItem]):
        x_lines, y_lines = self.get_grid_boundaries(cells)
        self.assign_grid_position(cells, x_lines, y_lines)
        self.assign_texts_to_cells(cells, texts)

        print("x_lines: ", x_lines)
        print("y_lines: ", y_lines)
        print("cells: ", cells)

        valid_cells = [
            c for c in cells
            if c.row is not None and c.col is not None 
            and c.row >= 0 and c.col >= 0
            and c.rowspan >= 1 and c.colspan >= 1
        ]
        if not valid_cells:
            return None

        # create a Table instance
        table = Table(
            id="1",
            bbox=BBox(x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]),
            table_rows=[],
        )
        
        # create TableRow instances and assign cells
        max_row = max(c.row for c in valid_cells)

        for i in range(max_row + 1):
            table_row = TableRow(
                row_id=i,
                bbox=BBox(x_lines[0], y_lines[i], x_lines[-1], y_lines[i+1]),
                cells=[],
            )
            for cell in valid_cells:
                if cell.row == i:
                    table_row.cells.append(cell)

            table_row.cells.sort(key=lambda c: c.col)
            table.table_rows.append(table_row)

        table = asdict(table)

        return table


    def assign_texts_to_cells(self, cells: list[Cell], texts: list[TextItem]):
        for text in texts:
            tx = (text.bbox.x_max + text.bbox.x_min) / 2
            ty = (text.bbox.y_max + text.bbox.y_min) / 2

            for cell in cells:
                if (cell.bbox.x_min <= tx <= cell.bbox.x_max and
                    cell.bbox.y_min <= ty <= cell.bbox.y_max):

                    cell.texts.append(text)
                    break

    def assign_grid_position(self, cells: list[Cell], x_lines: list, y_lines: list):
        for cell in cells:
            # find col_start and col_end index
            col_start = self.find_index(x_lines, cell.bbox.x_min)
            col_end = self.find_index(x_lines, cell.bbox.x_max)
            row_start = self.find_index(y_lines, cell.bbox.y_min)
            row_end = self.find_index(y_lines, cell.bbox.y_max)

            if None in (col_start, col_end, row_start, row_end):
                print("Error mapping col or row")
                continue

            if col_end <= col_start or row_end <= row_start:
                print("col_end or row_end has smaller value")
                continue

            # assign col, row, colspan, rowspan
            cell.col = col_start
            cell.colspan = col_end - col_start
            cell.row = row_start
            cell.rowspan = row_end - row_start  

    def get_grid_boundaries(self, cells: list[Cell]) -> tuple[list[float], list[float]]:
        # consolidate all x_min and x_max and sort them
        x_lines = []
        for c in cells:
            x_lines.append(c.bbox.x_max)
            x_lines.append(c.bbox.x_min)

        # consolidate all y_min and y_max and sort them
        y_lines = []
        for c in cells:
            y_lines.append(c.bbox.y_max)
            y_lines.append(c.bbox.y_min)

        # remove repeated values with the tolerance of 3px
        x_lines = self.merge_lines(x_lines)
        y_lines = self.merge_lines(y_lines)

        return x_lines, y_lines
    
    @staticmethod
    def merge_lines(lines, tolerance=10): # ★ 체인 현상 막았으니 10 정도로 타이트하게 가자
        if not lines:
            return []
        
        lines.sort()
        merged = []
        current_cluster = [lines[0]]
        
        for val in lines[1:]:
            # ★ 핵심 수정: [-1]이 아니라 [0]! 
            # 그룹의 '첫 시작점'을 기준으로 거리를 재야 선이 무한정 뚱뚱해지는 걸 막는다.
            if val - current_cluster[0] <= tolerance:
                current_cluster.append(val)
            else:
                merged.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [val]
                
        if current_cluster:
            merged.append(sum(current_cluster) / len(current_cluster))

        return merged
    
    @staticmethod
    def find_index(lines, value):
        # ★ 기존에 있던 tolerance 조건문 다 지우고, 무조건 가장 가까운 선을 잡게 만든 코드
        if not lines:
            return None
        return min(range(len(lines)), key=lambda i: abs(lines[i] - value))
    
    # @staticmethod
    # def merge_lines(lines, tolerance=10):
    #     if not lines:
    #         return []
        
    #     lines.sort()
    #     merged = [lines[0]]
        
    #     for val in lines[1:]:
    #         if abs(val - merged[-1]) > tolerance:
    #             merged.append(val)

    #     return merged
    
    # @staticmethod
    # def find_index(lines, value, tolerance=3):
    #     for i in range(len(lines)):
    #         if abs(value - lines[i]) < tolerance:
    #             return i
    #     return None




    

# ocr = PaddleOCR(
#         use_doc_orientation_classify=True, 
#         use_doc_unwarping=False, 
#         use_textline_orientation=False,
#         lang='korean') # text detection + text recognition
# ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True, lang='korean') # text image preprocessing + text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False) # Switch to PP-OCRv5_mobile models

# model = TextDetection(model_name="PP-OCRv5_server_det")

# structure = LayoutDetection(model_name="PP-DocLayout_plus-L")

# structure = PPStructureV3(use_doc_unwarping=False, lang='korean',)
# pipeline = PPStructureV3(lang="en") # Set the lang parameter to use the English text recognition model. For other supported languages, see Section 5: Appendix. By default, both Chinese and English text recognition models are enabled.
# pipeline = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# pipeline = PPStructureV3(device="gpu") # Use device to specify GPU for model inference