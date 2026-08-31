from fastapi import APIRouter, UploadFile, HTTPException, Depends
import numpy as np
import cv2
import math
from pathlib import Path
from sqlmodel import select, Session

# from src.services.table_extraction import TableExtraction
from ....db.session import get_session
from ....models.job_model import Job, JobResult, JobStatus
from ....util import now_toronto
from src.tasks.ocr_tasks import process_job

router = APIRouter()

def get_grid_boundaries(cells):
    # consolidate all x_min and x_max and sort them
    x_lines = []
    for c in cells:
        bbox = c["bbox"]
        x_lines.append(bbox["x_max"])
        x_lines.append(bbox["x_min"])

    # consolidate all y_min and y_max and sort them
    y_lines = []
    for c in cells:
        bbox = c["bbox"]
        y_lines.append(bbox["y_max"])
        y_lines.append(bbox["y_min"])

    return merge_lines(x_lines), merge_lines(y_lines)

def merge_lines(lines, tolerance=3):
    if not lines:
        return []
    
    lines.sort()
    merged = [lines[0]]
    
    for val in lines[1:]:
        if abs(val - merged[-1]) > tolerance:
            merged.append(val)

    return merged

@router.get("/{job_id}", tags=["jobs"], status_code=200)
async def get_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/{job_id}/result", tags=["jobs"])
async def get_result(job_id: int, session: Session = Depends(get_session)):

    # 1. 일단 대빵(Job) 테이블부터 조회해서 현재 진짜 상태를 가져옴
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job itself not found")

    # 2. 작업이 아직 안 끝났거나 실패했다면, 테이블 결과 파싱할 필요 없이 바로 상태만 리턴
    if job.status != JobStatus.done:
        return {
            "status": job.status, # "queued", "processing", "failed" 등 진짜 상태 전달
            "message": job.error_message if job.status == JobStatus.failed else "Processing..."
        }
    
    # result == {1 table}
    # TODO: 2 or more table in array
    data = session.exec(select(JobResult).where(JobResult.job_id == job_id)).first()

    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = data.result_json

    if not result:
        raise HTTPException(status_code=404, detail="Parsed result not found")
    
    cells = [cell for row in result["table_rows"] for cell in row["cells"]]

    x_lines, y_lines = get_grid_boundaries(cells)
    
    # Takes care of only 1 table, TODO: imporve so it takes care of 2 or more tables
    front_res = {
        "status": job.status,
        "tables": [
            {
                "id": result["id"],
                "bbox": result["bbox"],
                "x_lines": x_lines,
                "y_lines": y_lines,
                "cells": [
                    {
                        "row": cell["row"],
                        "col": cell["col"],
                        "rowspan": cell["rowspan"],
                        "colspan": cell["colspan"],
                        "value": " ".join(text["value"] for text in cell["texts"]) if cell["texts"] else "",
                        "score": min(text["score"] for text in cell["texts"]) if cell["texts"] else None,
                        "bbox": cell["bbox"]
                    }
                    for row in result["table_rows"]
                    for cell in row["cells"]
                ]
            }
        ]
    }

    return front_res    

@router.post("/", tags=["jobs"], status_code=201)
async def upload_job(file: UploadFile, session: Session = Depends(get_session)):
    content = await file.read()
    filename = file.filename
    file_type = file.content_type
    print(file)

    # create job instance
    job = Job(
        original_filename = filename,
        file_type = file_type,
    )

    # add job to db
    # session = next(get_session()) # use next() due to Generator used 
    session.add(job)
    session.commit()
    session.refresh(job)

    # save img file to local disk
    upload_dir = Path(f"/Users/sumin/Desktop/study/Personal Projects/ocr-document/storage/uploads/jobs/{job.id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        f.write(content)

    job.file_path = str(file_path)
    job.updated_at = now_toronto()
    session.commit()

    process_job.delay(job.id)

    # return job id, job status, response
    return {
        "job_id": job.id,
        "status": job.status
    }

@router.patch("/{job_id}/result", response_model=JobResult, tags=["jobs"])
def update_result(job_id: int, updated_result: dict):
    session = next(get_session())
    job_result = session.get(JobResult, job_id)
    if not job_result:
        raise HTTPException(status_code=404, detail="Job result not found")
    
    job_result.edited_json = updated_result
    session.add(job_result)
    session.commit()
    session.refresh(job_result)
    
    return job_result




# @router.post("/{job_id}/retry", tags=["jobs"])
# def retry_job():
#     # change status to queued and euqueue
#     pass

# @router.post("/{job_id}/cancel", tags=["jobs"])
# def cancel_job():
#     pass

# @router.post("/tem", tags=["jobs"])
# async def upload_file(file: UploadFile):
#     print(file)
#     content = await file.read()

#     nparr = np.frombuffer(content, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if image is None:
#         return {"status": "invalid image"}
    
#     texts, cells = table_structure.detect(image)

#     # metadata
#     meta = {
#         "file_name": file.filename,
#         "file_type": file.content_type,
#     }

#     # extract y_values
#     y_vals = []
#     for cell in cells:
#         y_min = math.floor(cell.bbox.y_min)
#         y_max = math.floor(cell.bbox.y_max)
#         y_vals.append({
#             "y_min": y_min,
#             "y_max": y_max,
#         })

#     # row boundaries
#     y_boundaries = []
#     y_first = y_vals[0]['y_min']
#     y_boundaries.append(y_first)
#     for y in y_vals:
#         y_boundaries.append(y['y_max'])

#     y_boundaries = list(set(y_boundaries))
#     y_boundaries.sort()

#     return {"status": "ok", "meta": meta, "texts": texts, "cells": cells, "y_boundaries": y_boundaries}