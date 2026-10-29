"""SQLAlchemy engine/session setup for the local SQLite database."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = None
SessionLocal: sessionmaker | None = None


def configure(database_url: str | None = None) -> None:
    """(Re)create the engine. Tests call this with a temporary database URL."""
    global engine, SessionLocal
    url = database_url or settings.resolved_database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from app.models import entities  # noqa: F401  (registers the tables)

    if engine is None:
        configure()
    Base.metadata.create_all(bind=engine)


def check_db() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db() -> Iterator[Session]:
    if SessionLocal is None:
        configure()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
