"""DB 연결 풀과 요청별 세션을 제공한다. 변경 내용의 commit은 호출자가 담당한다."""

import os

from sqlmodel import Session, create_engine

engine = create_engine(os.getenv("DATABASE_URL"))


def get_session() -> Session:
    """API 요청에 세션을 제공하고 요청이 끝나면 닫는다."""
    with Session(engine) as session:
        yield session
