"""Alert system — configurable fatigue threshold alerts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..models.fatigue import FatigueScore


@dataclass
class Alert:
    """A fatigue alert for a specific player."""

    player_id: int
    level: str                 # "moderate", "high", "critical"
    score: float
    message: str
    timestamp: float
    contributing_factors: dict[str, float]

    def to_dict(self) -> dict[str, str]:
        return {
            "player_id": str(self.player_id),
            "level": self.level,
            "score": str(round(self.score, 1)),
            "message": self.message,
            "timestamp": str(self.timestamp),
            "factors": str(self.contributing_factors),
        }


class AlertManager:
    """Manages fatigue alerts with cooldowns to prevent alert spam."""

    def __init__(
        self,
        thresholds: dict[str, int] | None = None,
        cooldown: int = 120,
    ) -> None:
        defaults = {"moderate": 55, "high": 75, "critical": 90}
        self.thresholds = thresholds or defaults
        self.cooldown = cooldown  # seconds
        self._last_alert: dict[tuple[int, str], float] = {}

    def check(self, scores: dict[int, FatigueScore]) -> list[Alert]:
        """Check all player scores against thresholds."""
        alerts: list[Alert] = []
        now = time.time()

        for pid, score in scores.items():
            for level in ["critical", "high", "moderate"]:
                threshold = self.thresholds.get(level, 100)
                if score.score >= threshold:
                    key = (pid, level)
                    last = self._last_alert.get(key, 0)
                    if now - last >= self.cooldown:
                        alert = Alert(
                            player_id=pid,
                            level=level,
                            score=score.score,
                            message=f"Player {pid} fatigue {level}: {score.score:.0f}/100",
                            timestamp=now,
                            contributing_factors=score.contributing_factors,
                        )
                        alerts.append(alert)
                        self._last_alert[key] = now
                    break  # Only alert at highest triggered level

        return alerts
