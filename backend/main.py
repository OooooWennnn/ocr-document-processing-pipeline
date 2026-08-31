from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.db.engine import engine
from sqlmodel import SQLModel
import src.models.job_model

from src.api.v1.api import api_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

SQLModel.metadata.create_all(engine)




