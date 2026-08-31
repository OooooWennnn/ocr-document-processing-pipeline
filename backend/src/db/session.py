from .engine import engine
from sqlmodel import Session, SQLModel

SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    with Session(engine) as session:
        yield session