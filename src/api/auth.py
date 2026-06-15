"""API authentication — API key validation with team isolation."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import APIKey, Team, get_session

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_api_key(key: str) -> str:
    """SHA-256 hash of an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a new API key."""
    return f"ss_{secrets.token_urlsafe(32)}"


async def get_current_team(
    api_key: str | None = Security(api_key_header),
    session: AsyncSession = Depends(get_session),
) -> Team:
    """Validate API key and return the associated team."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hash_api_key(api_key)
    result = await session.execute(
        select(APIKey)
        .where(APIKey.key_hash == key_hash, APIKey.is_active.is_(True))
    )
    api_key_record = result.scalar_one_or_none()

    if not api_key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if api_key_record.expires_at and api_key_record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="API key expired")

    result = await session.execute(
        select(Team).where(Team.id == api_key_record.team_id, Team.is_active.is_(True))
    )
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(status_code=401, detail="Team not found or inactive")

    return team


def require_role(required_role: str):
    """Dependency that checks the API key's role."""
    async def check_role(
        api_key: str | None = Security(api_key_header),
        session: AsyncSession = Depends(get_session),
    ) -> Team:
        team = await get_current_team(api_key, session)

        key_hash = hash_api_key(api_key)
        result = await session.execute(
            select(APIKey).where(APIKey.key_hash == key_hash)
        )
        key_record = result.scalar_one()

        role_hierarchy = {"viewer": 0, "analyst": 1, "admin": 2}
        if role_hierarchy.get(key_record.role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Requires {required_role} role or higher",
            )

        return team

    return check_role
