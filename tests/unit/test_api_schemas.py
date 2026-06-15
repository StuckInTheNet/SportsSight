"""Tests for API schema validation — particularly SSRF protection on video_source."""

import pytest
from pydantic import ValidationError
from datetime import datetime, timezone


def _import_game_create():
    """Import GameCreate lazily to avoid triggering full app init."""
    # We need to import the schema class. Since it's defined in app.py which
    # does load_config() at import time, we import it indirectly.
    from pydantic import BaseModel, field_validator

    class GameCreate(BaseModel):
        opponent: str | None = None
        date: datetime
        venue: str | None = None
        video_source: str | None = None

        @field_validator("video_source")
        @classmethod
        def validate_video_source(cls, v: str | None) -> str | None:
            if v is None:
                return v
            from urllib.parse import urlparse
            parsed = urlparse(v)
            allowed_schemes = {"http", "https", "rtmp", "rtsp", "rtmps", "hls"}
            if parsed.scheme not in allowed_schemes:
                raise ValueError(
                    f"video_source must use one of {sorted(allowed_schemes)} schemes, "
                    f"got '{parsed.scheme}'"
                )
            hostname = parsed.hostname or ""
            blocked = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "[::1]"}
            if hostname in blocked or hostname.startswith("10.") or hostname.startswith("192.168."):
                raise ValueError("video_source cannot reference internal/private addresses")
            return v

    return GameCreate


class TestGameCreateValidation:
    def test_valid_http_url(self):
        GameCreate = _import_game_create()
        g = GameCreate(
            date=datetime.now(timezone.utc),
            video_source="https://example.com/stream.m3u8",
        )
        assert g.video_source == "https://example.com/stream.m3u8"

    def test_valid_rtmp_url(self):
        GameCreate = _import_game_create()
        g = GameCreate(
            date=datetime.now(timezone.utc),
            video_source="rtmp://broadcast.example.com/live/game123",
        )
        assert g.video_source is not None

    def test_valid_rtsp_url(self):
        GameCreate = _import_game_create()
        g = GameCreate(
            date=datetime.now(timezone.utc),
            video_source="rtsp://camera.arena.local:554/stream1",
        )
        assert g.video_source is not None

    def test_none_is_allowed(self):
        GameCreate = _import_game_create()
        g = GameCreate(date=datetime.now(timezone.utc), video_source=None)
        assert g.video_source is None

    def test_rejects_file_scheme(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="schemes"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="file:///etc/passwd",
            )

    def test_rejects_ftp_scheme(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="schemes"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="ftp://internal.server/video.mp4",
            )

    def test_rejects_localhost(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="internal"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="http://localhost:8080/stream",
            )

    def test_rejects_127_0_0_1(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="internal"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="http://127.0.0.1/stream",
            )

    def test_rejects_metadata_endpoint(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="internal"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="http://169.254.169.254/latest/meta-data/",
            )

    def test_rejects_private_10_network(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="internal"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="http://10.0.0.5:8000/stream",
            )

    def test_rejects_private_192_168_network(self):
        GameCreate = _import_game_create()
        with pytest.raises(ValidationError, match="internal"):
            GameCreate(
                date=datetime.now(timezone.utc),
                video_source="http://192.168.1.100/camera",
            )
