"""Internal OCR data types, separate from database and HTTP models."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BBox:
    """Pixel bounds on the source page; x increases right and y increases down."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class TextItem:
    """Recognized text, bounds and confidence; None means no OCR score is available."""

    value: str
    score: float | None
    bbox: BBox


@dataclass
class Cell:
    """Cell bounds, grid position, row/column spans and assigned text."""

    row: int
    col: int
    bbox: BBox
    rowspan: int = 1
    colspan: int = 1
    texts: List[TextItem] = field(default_factory=list)


@dataclass
class TableRow:
    """Group cells that start on the same row."""

    row_id: int
    bbox: BBox
    cells: List[Cell] = field(default_factory=list)  # Key cells by column.


@dataclass
class Table:
    """A reconstructed table with an ID, bounds and rows."""

    id: str
    bbox: BBox
    table_rows: List[TableRow] = field(default_factory=list)
