"""Tests for the alert system."""

import time

import pytest

from src.models.fatigue import FatigueScore
from src.realtime.alerts import AlertManager


def _make_score(player_id: int, score: float, **kwargs) -> FatigueScore:
    return FatigueScore(
        player_id=player_id,
        timestamp_ms=0,
        score=score,
        confidence=0.9,
        trend="stable",
        baseline_deviation=0.1,
        **kwargs,
    )


class TestAlertManager:
    def test_no_alerts_below_threshold(self):
        mgr = AlertManager(thresholds={"moderate": 55, "high": 75, "critical": 90})
        scores = {1: _make_score(1, 30)}
        alerts = mgr.check(scores)
        assert len(alerts) == 0

    def test_moderate_alert(self):
        mgr = AlertManager(thresholds={"moderate": 55, "high": 75, "critical": 90})
        scores = {1: _make_score(1, 60)}
        alerts = mgr.check(scores)
        assert len(alerts) == 1
        assert alerts[0].level == "moderate"

    def test_critical_overrides_lower(self):
        mgr = AlertManager(thresholds={"moderate": 55, "high": 75, "critical": 90})
        scores = {1: _make_score(1, 95)}
        alerts = mgr.check(scores)
        assert len(alerts) == 1
        assert alerts[0].level == "critical"

    def test_cooldown_prevents_spam(self):
        mgr = AlertManager(
            thresholds={"moderate": 55, "high": 75, "critical": 90},
            cooldown=60,
        )
        scores = {1: _make_score(1, 60)}

        alerts1 = mgr.check(scores)
        assert len(alerts1) == 1

        # Immediate second check — should be suppressed
        alerts2 = mgr.check(scores)
        assert len(alerts2) == 0

    def test_multiple_players(self):
        mgr = AlertManager(thresholds={"moderate": 55, "high": 75, "critical": 90})
        scores = {
            1: _make_score(1, 60),
            2: _make_score(2, 80),
            3: _make_score(3, 40),
        }
        alerts = mgr.check(scores)
        assert len(alerts) == 2
        levels = {a.player_id: a.level for a in alerts}
        assert levels[1] == "moderate"
        assert levels[2] == "high"

    def test_alert_to_dict(self):
        mgr = AlertManager(thresholds={"moderate": 55})
        scores = {1: _make_score(1, 60, contributing_factors={"speed": 0.15})}
        alerts = mgr.check(scores)
        d = alerts[0].to_dict()
        assert "player_id" in d
        assert "level" in d
        assert "score" in d
