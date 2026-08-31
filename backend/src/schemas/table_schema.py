from pydantic import BaseModel
from datatime import datetime
from typing import Optional

class Cell(BaseModel):
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str
    bbox: Optional[list[int]]


class Table(BaseModel):
    id: str
    n_rows: int
    n_cols: int
    cells: list[Cell]
    bbox: Optional[list[int]]


class Meta(BaseModel):
    file_name: str
    file_type: str
    page_index: int = 0
    processed_at: datetime
    width: Optional[int] = None
    height: Optional[int] = None

class DocumentResult(BaseModel):
    meta : Meta
    tables: list[Table]