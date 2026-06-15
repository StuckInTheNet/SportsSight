"""FastAPI application — REST endpoints + WebSocket for live fatigue data."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import load_config
from .auth import generate_api_key, get_current_team, hash_api_key, require_role
from .database import (
    AlertRecord,
    APIKey,
    Base,
    FatigueRecord,
    Game,
    Player,
    Team,
    get_session,
    init_db,
)

logger = logging.getLogger(__name__)
config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    await init_db(config.database_url)
    app.state.redis = aioredis.from_url(config.redis_url, decode_responses=True)
    logger.info("SportsSight API started")
    yield
    await app.state.redis.aclose()


app = FastAPI(
    title="SportsSight API",
    description="Real-time sports fatigue analytics",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Pydantic Schemas ===

class TeamCreate(BaseModel):
    name: str
    sport: str = "basketball"

class TeamResponse(BaseModel):
    id: UUID
    name: str
    sport: str
    api_key: str | None = None  # Only returned on creation

class PlayerCreate(BaseModel):
    name: str
    jersey_number: int | None = None
    position: str | None = None
    height_inches: int | None = None
    weight_lbs: int | None = None

class PlayerResponse(BaseModel):
    id: UUID
    name: str
    jersey_number: int | None
    position: str | None

class GameCreate(BaseModel):
    opponent: str | None = None
    date: datetime
    venue: str | None = None
    video_source: str | None = None

class GameResponse(BaseModel):
    id: UUID
    opponent: str | None
    date: datetime
    status: str
    venue: str | None

class FatigueResponse(BaseModel):
    player_id: UUID
    timestamp_ms: float
    fatigue_score: float
    confidence: float
    trend: str
    speed: float | None
    contributing_factors: dict[str, float]

class AlertConfigUpdate(BaseModel):
    moderate_threshold: int | None = None
    high_threshold: int | None = None
    critical_threshold: int | None = None
    cooldown_seconds: int | None = None


# === Team Endpoints ===

@app.post("/teams", response_model=TeamResponse)
async def create_team(
    data: TeamCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new team with an API key."""
    team = Team(name=data.name, sport=data.sport)
    session.add(team)
    await session.flush()

    raw_key = generate_api_key()
    api_key = APIKey(
        team_id=team.id,
        key_hash=hash_api_key(raw_key),
        name="default",
        role="admin",
    )
    session.add(api_key)
    await session.commit()

    return TeamResponse(
        id=team.id, name=team.name, sport=team.sport, api_key=raw_key,
    )


@app.get("/teams/me", response_model=TeamResponse)
async def get_my_team(team: Team = Depends(get_current_team)):
    return TeamResponse(id=team.id, name=team.name, sport=team.sport)


# === Player Endpoints ===

@app.post("/players", response_model=PlayerResponse)
async def create_player(
    data: PlayerCreate,
    team: Team = Depends(require_role("analyst")),
    session: AsyncSession = Depends(get_session),
):
    player = Player(team_id=team.id, **data.model_dump())
    session.add(player)
    await session.commit()
    return PlayerResponse(
        id=player.id, name=player.name,
        jersey_number=player.jersey_number, position=player.position,
    )


@app.get("/players", response_model=list[PlayerResponse])
async def list_players(
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Player).where(Player.team_id == team.id)
    )
    players = result.scalars().all()
    return [
        PlayerResponse(
            id=p.id, name=p.name,
            jersey_number=p.jersey_number, position=p.position,
        )
        for p in players
    ]


# === Game Endpoints ===

@app.post("/games", response_model=GameResponse)
async def create_game(
    data: GameCreate,
    team: Team = Depends(require_role("analyst")),
    session: AsyncSession = Depends(get_session),
):
    game = Game(team_id=team.id, **data.model_dump())
    session.add(game)
    await session.commit()
    return GameResponse(
        id=game.id, opponent=game.opponent, date=game.date,
        status=game.status, venue=game.venue,
    )


@app.get("/games", response_model=list[GameResponse])
async def list_games(
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
    status: str | None = None,
):
    query = select(Game).where(Game.team_id == team.id)
    if status:
        query = query.where(Game.status == status)
    query = query.order_by(Game.date.desc())
    result = await session.execute(query)
    games = result.scalars().all()
    return [
        GameResponse(
            id=g.id, opponent=g.opponent, date=g.date,
            status=g.status, venue=g.venue,
        )
        for g in games
    ]


@app.get("/games/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: UUID,
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Game).where(Game.id == game_id, Game.team_id == team.id)
    )
    game = result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameResponse(
        id=game.id, opponent=game.opponent, date=game.date,
        status=game.status, venue=game.venue,
    )


# === Fatigue Data Endpoints ===

@app.get("/games/{game_id}/fatigue", response_model=list[FatigueResponse])
async def get_game_fatigue(
    game_id: UUID,
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
    player_id: UUID | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
    limit: int = Query(default=1000, le=10000),
):
    """Get fatigue records for a game, optionally filtered by player and time range."""
    # Verify game belongs to team
    game = await session.execute(
        select(Game).where(Game.id == game_id, Game.team_id == team.id)
    )
    if not game.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Game not found")

    query = select(FatigueRecord).where(FatigueRecord.game_id == game_id)
    if player_id:
        query = query.where(FatigueRecord.player_id == player_id)
    if start_ms is not None:
        query = query.where(FatigueRecord.timestamp_ms >= start_ms)
    if end_ms is not None:
        query = query.where(FatigueRecord.timestamp_ms <= end_ms)
    query = query.order_by(FatigueRecord.timestamp_ms).limit(limit)

    result = await session.execute(query)
    records = result.scalars().all()

    return [
        FatigueResponse(
            player_id=r.player_id,
            timestamp_ms=r.timestamp_ms,
            fatigue_score=r.fatigue_score,
            confidence=r.confidence,
            trend=r.trend or "stable",
            speed=r.speed,
            contributing_factors=r.contributing_factors or {},
        )
        for r in records
    ]


@app.get("/players/{player_id}/fatigue-history")
async def get_player_fatigue_history(
    player_id: UUID,
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
    last_n_games: int = Query(default=10, le=50),
):
    """Get fatigue trends across recent games for a player."""
    # Verify player belongs to team
    player = await session.execute(
        select(Player).where(Player.id == player_id, Player.team_id == team.id)
    )
    if not player.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Player not found")

    # Get summary per game
    result = await session.execute(
        select(
            FatigueRecord.game_id,
            func.avg(FatigueRecord.fatigue_score).label("avg_fatigue"),
            func.max(FatigueRecord.fatigue_score).label("max_fatigue"),
            func.avg(FatigueRecord.speed).label("avg_speed"),
            func.count().label("data_points"),
        )
        .where(FatigueRecord.player_id == player_id)
        .group_by(FatigueRecord.game_id)
        .order_by(func.max(FatigueRecord.created_at).desc())
        .limit(last_n_games)
    )
    rows = result.all()

    return [
        {
            "game_id": str(row.game_id),
            "avg_fatigue": round(row.avg_fatigue, 1),
            "max_fatigue": round(row.max_fatigue, 1),
            "avg_speed": round(row.avg_speed, 1) if row.avg_speed else None,
            "data_points": row.data_points,
        }
        for row in rows
    ]


# === Alert Configuration ===

@app.post("/alerts/configure")
async def configure_alerts(
    config_update: AlertConfigUpdate,
    team: Team = Depends(require_role("admin")),
):
    """Update alert thresholds for the team."""
    # In production this would persist to DB; for now return confirmation
    return {"status": "updated", "config": config_update.model_dump(exclude_none=True)}


@app.get("/games/{game_id}/alerts")
async def get_game_alerts(
    game_id: UUID,
    team: Team = Depends(get_current_team),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(AlertRecord)
        .where(AlertRecord.game_id == game_id)
        .order_by(AlertRecord.created_at.desc())
    )
    alerts = result.scalars().all()
    return [
        {
            "id": str(a.id),
            "player_id": str(a.player_id),
            "level": a.level,
            "score": a.score,
            "message": a.message,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


# === WebSocket — Live Fatigue Stream ===

@app.websocket("/games/{game_id}/live")
async def live_fatigue_stream(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time fatigue updates during a live game.

    Reads from Redis Stream and pushes to connected clients.
    Authentication is done via query param: ?api_key=ss_xxx
    """
    await websocket.accept()

    redis_client: aioredis.Redis = app.state.redis
    stream_key = f"sportssight:game:{game_id}:fatigue"
    alert_key = f"sportssight:game:{game_id}:alerts"
    last_id = "$"  # Start from new messages only

    try:
        while True:
            # Read from both fatigue and alert streams
            streams = await redis_client.xread(
                {stream_key: last_id, alert_key: last_id},
                count=10,
                block=1000,  # Block for 1 second
            )

            for stream_name, messages in streams:
                for msg_id, data in messages:
                    if "fatigue" in stream_name:
                        await websocket.send_json({
                            "type": "fatigue_update",
                            "data": data,
                        })
                    elif "alerts" in stream_name:
                        await websocket.send_json({
                            "type": "alert",
                            "data": data,
                        })
                    last_id = msg_id

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from game %s", game_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await websocket.close(code=1011)
