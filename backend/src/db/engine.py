"""Create the connection pool from DATABASE_URL."""

import os
from sqlmodel import create_engine

db_url = os.getenv("DATABASE_URL")

engine = create_engine(db_url)
