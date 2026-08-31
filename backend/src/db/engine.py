import os
from dotenv import load_dotenv
from sqlmodel import create_engine

load_dotenv()

db_url = os.getenv('DATABASE_URL')

engine = create_engine(db_url)