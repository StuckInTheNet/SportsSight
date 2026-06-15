"""Tests for database manager and session factory."""

import pytest

from src.api.database import DatabaseManager, db


class TestDatabaseManager:
    def test_not_initialized_by_default(self):
        mgr = DatabaseManager()
        assert mgr.is_initialized is False

    def test_session_raises_when_not_initialized(self):
        mgr = DatabaseManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            mgr.session()

    def test_default_instance_exists(self):
        """The module-level `db` instance should exist."""
        assert isinstance(db, DatabaseManager)
