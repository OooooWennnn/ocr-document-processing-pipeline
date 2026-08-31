from sqlmodel import Session
import cv2

from src.core.celery_app import celery_app as app
from src.db.engine import engine
from src.models.job_model import Job, JobResult, JobStatus
from src.util import now_toronto
from src.services.table_extraction import TableExtraction

table_extraction = None
print("ocr_tasks imported")

@app.task
def process_job(job_id: int):
    print(f"process_job task loaded for Job ID: {job_id}")
    # 1. get job from db using job_id
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            print(f"Job {job_id} not found")
            return
        
        try:
            # 2. update status=processing
            job.status = JobStatus.processing
            session.commit()

            # 3. process ocr
            image = cv2.imread(job.file_path)
            if image is None:
                job.error_message = "Image is invalid"
                job.status = JobStatus.failed
                session.commit()
                return
            
            # load model if not loaded
            global table_extraction
            if table_extraction is None:
                table_extraction = TableExtraction()

            raw_result, result = table_extraction.detect(image)
            if raw_result is None or result is None:
                job.error_message = "OCR process failed"
                job.status = JobStatus.failed
                session.commit()
                return

            # 4. create job_result instance
            job_result = JobResult(
                job_id = job_id,
                raw_json = raw_result,
                result_json = result,
            )

            # 5. update status=done
            job.status = JobStatus.done
            job.updated_at = now_toronto()

            # 6. save job_result to db
            session.add(job_result)
            session.commit()

        except Exception as e:
            session.rollback()
            job.status = JobStatus.failed
            job.error_message = str(e)
            session.commit()