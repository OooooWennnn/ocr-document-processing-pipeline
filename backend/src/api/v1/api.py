"""Mount job endpoints under /api/v1."""

from fastapi import APIRouter
from .endpoints import jobs

api_router = APIRouter()


# /api/v1/
@api_router.get("/")

async def root():
    """Return a response from the API root."""
    return {"Hello": "World"}


@api_router.get("/health")
def health_check():
    """Legacy placeholder; Docker uses main.py /health."""
    pass


# /api/v1/jobs
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
