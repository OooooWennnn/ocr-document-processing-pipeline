from fastapi import APIRouter
from .endpoints import jobs

api_router = APIRouter()

# /api/v1/
@api_router.get("/")
async def root():
    return {"Hello": "World"}

@api_router.get("/health")
def health_check():
    pass

# /api/v1/jobs
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
