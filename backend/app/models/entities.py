"""Database tables.

Only metadata lives in SQLite. Images, CSVs and checkpoints stay on disk and the
tables store their paths.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRun(Base):
    """One executed analysis (any tutorial module). Powers the history view and reports."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    module: Mapped[str] = mapped_column(String(64), index=True)
    tutorial: Mapped[str | None] = mapped_column(String(8), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    dataset: Mapped[str | None] = mapped_column(String(100), nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Prediction(Base):
    """A pneumonia screening prediction made on an uploaded X-ray."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[str] = mapped_column(String(32))
    probability_pneumonia: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UploadedDataset(Base):
    """A tabular file uploaded by the user through the EHR explorer."""

    __tablename__ = "uploaded_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(Text)
    rows: Mapped[int] = mapped_column(Integer)
    columns: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
