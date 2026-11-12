"""SQLAlchemy 2 models for PRD section 6.1."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nickname: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class GameSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    ip_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_canary: Mapped[bool] = mapped_column(Boolean, default=False)


class Level(Base):
    __tablename__ = "levels"
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    level: Mapped[int] = mapped_column(Integer, primary_key=True)
    secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    messages_used: Mapped[int] = mapped_column(Integer, default=0)
    # v1.1 (Alembic 0002, CONTRACTS 7.5)
    skipped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    points: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    hints_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    level: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text)
    guard_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(8), nullable=True)
    category: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    shadow_verdict: Mapped[str | None] = mapped_column(String(8), nullable=True)
    shadow_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    agent_reply: Mapped[str | None] = mapped_column(Text, nullable=True)
    leaked: Mapped[bool] = mapped_column(Boolean, default=False)
    matched_variant: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Label(Base):
    __tablename__ = "labels"
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"), primary_key=True)
    label: Mapped[str] = mapped_column(String(8), nullable=False)
    category: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(8), nullable=False)  # leak|judge|human
    judge_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    judge_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hf_repo: Mapped[str | None] = mapped_column(String(128), nullable=True)
    alias: Mapped[str | None] = mapped_column(String(16), nullable=True)
    gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
