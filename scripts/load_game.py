"""Load processed game results into the database and replay through Redis.

Usage:
    python scripts/load_game.py data/processed/nba_720p_reid_results.json

This:
1. Creates a team + API key if none exists
2. Creates player records for each tracked player
3. Inserts fatigue records into the database
4. Replays the timeline through Redis Streams (for dashboard WebSocket)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import asyncpg
import redis.asyncio as aioredis
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL_SYNC", "postgresql://sportssight:sportssight@localhost:5432/sportssight")
ASYNC_DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://sportssight:sportssight@localhost:5432/sportssight")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
API_SECRET = os.getenv("API_SECRET_KEY", "sportssight-dev-secret-2026")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_game.py <results.json>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"File not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    game_id = data["game_id"]
    timeline = data["timeline"]
    print(f"Game: {game_id}, {len(timeline)} frames")

    # Collect unique player IDs
    all_pids = set()
    for entry in timeline:
        all_pids.update(entry["scores"].keys())
    print(f"Unique players: {len(all_pids)}")

    # Connect to database
    conn = await asyncpg.connect(DATABASE_URL)
    print("Connected to database")

    # Create team
    team_id = str(uuid4())
    await conn.execute(
        "INSERT INTO teams (id, name, sport, is_active) VALUES ($1, $2, $3, $4) ON CONFLICT (name) DO NOTHING",
        team_id, "SportsSight Demo", "basketball", True,
    )
    # Get the actual team_id (might already exist)
    row = await conn.fetchrow("SELECT id FROM teams WHERE name = $1", "SportsSight Demo")
    team_id = str(row["id"])
    print(f"Team ID: {team_id}")

    # Create API key
    import hashlib, secrets
    raw_key = f"ss_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    await conn.execute(
        "INSERT INTO api_keys (id, team_id, key_hash, name, role, is_active) VALUES ($1, $2, $3, $4, $5, $6)",
        str(uuid4()), team_id, key_hash, "demo", "admin", True,
    )
    print(f"API Key: {raw_key}")
    print(f"  (save this — needed for dashboard auth)")

    # Create players
    pid_to_uuid: dict[str, str] = {}
    for pid in sorted(all_pids, key=lambda x: int(x)):
        player_uuid = str(uuid4())
        pid_to_uuid[pid] = player_uuid
        await conn.execute(
            "INSERT INTO players (id, team_id, name, jersey_number) VALUES ($1, $2, $3, $4)",
            player_uuid, team_id, f"Player {pid}", int(pid) if int(pid) < 100 else None,
        )
    print(f"Created {len(pid_to_uuid)} players")

    # Create game
    game_uuid = str(uuid4())
    from datetime import datetime, timezone
    await conn.execute(
        "INSERT INTO games (id, team_id, opponent, date, status) VALUES ($1, $2, $3, $4, $5)",
        game_uuid, team_id, "Opponent", datetime.now(timezone.utc), "completed",
    )
    print(f"Game UUID: {game_uuid}")

    # Insert fatigue records (batch for speed)
    print("Inserting fatigue records...")
    batch = []
    for entry in timeline:
        for pid, score in entry["scores"].items():
            if pid not in pid_to_uuid:
                continue
            batch.append((
                str(uuid4()),
                game_uuid,
                pid_to_uuid[pid],
                float(entry["timestamp_ms"]),
                int(entry["frame"]),
                float(score["score"]),
                float(score["confidence"]),
                score.get("trend", "stable"),
                float(score.get("contributing_factors", {}).get("speed", 0)) * 100 if score.get("contributing_factors") else None,
                json.dumps(score.get("contributing_factors", {})),
            ))

            if len(batch) >= 5000:
                await conn.executemany(
                    """INSERT INTO fatigue_records
                       (id, game_id, player_id, timestamp_ms, frame_number,
                        fatigue_score, confidence, trend, speed, contributing_factors)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)""",
                    batch,
                )
                print(f"  Inserted {len(batch)} records...")
                batch.clear()

    if batch:
        await conn.executemany(
            """INSERT INTO fatigue_records
               (id, game_id, player_id, timestamp_ms, frame_number,
                fatigue_score, confidence, trend, speed, contributing_factors)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)""",
            batch,
        )
        print(f"  Inserted final {len(batch)} records")

    total = await conn.fetchval("SELECT count(*) FROM fatigue_records WHERE game_id = $1", game_uuid)
    print(f"Total fatigue records: {total}")
    await conn.close()

    # Replay through Redis for WebSocket consumers
    print(f"\nReplaying {len(timeline)} frames through Redis Streams...")
    print("  (Connect dashboard to see live updates)")
    print(f"  Game ID for WebSocket: {game_uuid}")

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    stream_key = f"sportssight:game:{game_uuid}:fatigue"

    replay_speed = 10  # 10x speed
    prev_ts = 0.0
    for i, entry in enumerate(timeline):
        ts = entry["timestamp_ms"]
        if prev_ts > 0:
            delay = (ts - prev_ts) / 1000.0 / replay_speed
            if delay > 0:
                await asyncio.sleep(delay)
        prev_ts = ts

        payload = {
            "game_id": game_uuid,
            "frame": str(entry["frame"]),
            "timestamp_ms": str(ts),
            "scores": json.dumps(entry["scores"]),
        }
        await redis_client.xadd(stream_key, payload, maxlen=10000)

        if (i + 1) % 500 == 0:
            elapsed_game = ts / 1000 / 60
            print(f"  Replayed {i+1}/{len(timeline)} frames ({elapsed_game:.1f} min game time)")

    print(f"\nDone! Replay complete.")
    print(f"\n{'='*50}")
    print(f"Dashboard connection info:")
    print(f"  API URL:    http://localhost:8000")
    print(f"  API Key:    {raw_key}")
    print(f"  Game ID:    {game_uuid}")
    print(f"  WebSocket:  ws://localhost:8000/games/{game_uuid}/live?api_key={raw_key}")
    print(f"{'='*50}")

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
