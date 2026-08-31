from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    

@dataclass
class TextItem:
    """
    Represents a detected text

    Attributes:
        text: 
        confidence:
        bbox:
    """
    value: str
    score: float
    bbox: BBox
    
@dataclass
class Cell:
    """
    Represents a cell in a table 
    """
    row: int
    col: int
    bbox: BBox
    rowspan: int = 1
    colspan: int = 1
    texts: List[TextItem] = field(default_factory=list)

@dataclass
class TableRow:
    """
    Represents a row in a table
    """
    row_id: int
    bbox: BBox
    cells: List[Cell] = field(default_factory=dict) # key = col

@dataclass
class RawTable:
    """
    Represents a raw table with detected cells but without row/col assignment
    """
    id: str
    bbox: BBox
    cells: List[Cell] = field(default_factory=list)

@dataclass
class Table:
    """
    Represents a table
    """
    id: str
    bbox: BBox
    table_rows: List[TableRow] = field(default_factory=list)

@dataclass
class DocRegion:
    pass
