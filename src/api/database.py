"""Database models and session management — PostgreSQL + TimescaleDB."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# === Multi-tenant Models ===

class Team(Base):
    __tablename__ = "teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    sport = Column(String(50), default="basketball")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    api_keys = relationship("APIKey", back_populates="team")
    players = relationship("Player", back_populates="team")
    games = relationship("Game", back_populates="team")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    key_hash = Column(String(128), nullable=False, unique=True)
    name = Column(String(100))
    role = Column(String(20), default="viewer")  # admin, analyst, viewer
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    team = relationship("Team", back_populates="api_keys")


class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    name = Column(String(100), nullable=False)
    jersey_number = Column(Integer, nullable=True)
    position = Column(String(20), nullable=True)
    height_inches = Column(Integer, nullable=True)
    weight_lbs = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    team = relationship("Team", back_populates="players")
    fatigue_records = relationship("FatigueRecord", back_populates="player")


class Game(Base):
    __tablename__ = "games"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    opponent = Column(String(100), nullable=True)
    date = Column(DateTime(timezone=True), nullable=False)
    venue = Column(String(200), nullable=True)
    status = Column(String(20), default="pending")  # pending, live, completed
    video_source = Column(String(500), nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    team = relationship("Team", back_populates="games")
    fatigue_records = relationship("FatigueRecord", back_populates="game")


class FatigueRecord(Base):
    """Time-series fatigue data — one row per player per time step."""
    __tablename__ = "fatigue_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    timestamp_ms = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=False)
    fatigue_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    trend = Column(String(20))
    speed = Column(Float)
    acceleration = Column(Float)
    jump_height = Column(Float)
    defensive_stance = Column(Float)
    torso_lean = Column(Float)
    court_x = Column(Float)
    court_y = Column(Float)
    contributing_factors = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    game = relationship("Game", back_populates="fatigue_records")
    player = relationship("Player", back_populates="fatigue_records")

    __table_args__ = (
        Index("idx_fatigue_game_player_time", "game_id", "player_id", "timestamp_ms"),
        Index("idx_fatigue_game_time", "game_id", "timestamp_ms"),
    )


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_id = Column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    player_id = Column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    level = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    message = Column(Text)
    contributing_factors = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# === Database Manager ===


class DatabaseManager:
    """Encapsulates engine and session factory — no module-level globals."""

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def is_initialized(self) -> bool:
        return self._engine is not None

    async def init(self, database_url: str) -> None:
        """Initialize the async database engine and session factory."""
        self._engine = create_async_engine(
            database_url, echo=False, pool_size=20, max_overflow=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    def session(self) -> AsyncSession:
        """Create a new session (caller manages lifecycle)."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._session_factory()


# Default instance — used by the FastAPI app
db = DatabaseManager()


async def init_db(database_url: str) -> None:
    """Initialize the default database manager."""
    await db.init(database_url)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session with proper rollback on error."""
    session = db.session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_raw_session() -> AsyncSession:
    """Get a raw session for use outside FastAPI DI (e.g. WebSocket auth)."""
    return db.session()
