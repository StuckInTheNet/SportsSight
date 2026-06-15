"""Integration test — end-to-end pipeline with synthetic data."""

import numpy as np
import pytest

from src.features.extractor import FeatureExtractor, PlayerFeatures
from src.models.fatigue import FatigueModel, FatigueScore
from src.realtime.alerts import AlertManager


class TestEndToEndFatiguePipeline:
    """Tests the feature extraction → fatigue scoring → alert pipeline
    using synthetic data (no GPU / no video required)."""

    def test_full_pipeline_synthetic(self):
        """Simulate a player going from fresh to fatigued."""
        extractor = FeatureExtractor(fps=15)
        fatigue_model = FatigueModel(baseline_window_minutes=0.1)
        alert_mgr = AlertManager(
            thresholds={"moderate": 55, "high": 75, "critical": 90},
            cooldown=0,  # No cooldown for test
        )

        all_scores: list[FatigueScore] = []
        all_alerts = []

        # Simulate 20 seconds of play
        for frame in range(300):
            timestamp_ms = frame * 66.67  # ~15fps

            # Player gradually slows down
            decay = min(1.0, frame / 300)
            features = {
                1: PlayerFeatures(
                    player_id=1,
                    timestamp_ms=timestamp_ms,
                    frame_number=frame,
                    speed=20.0 * (1 - decay * 0.6),
                    acceleration=10.0 * (1 - decay * 0.5),
                    jump_height=2.0 * (1 - decay * 0.4),
                    defensive_stance_depth=160 - decay * 30,
                    torso_lean=5 + decay * 15,
                ),
            }

            scores = fatigue_model.update(features)
            alerts = alert_mgr.check(scores)

            all_scores.append(scores[1])
            all_alerts.extend(alerts)

        # Fatigue should generally increase over time
        early_avg = np.mean([s.score for s in all_scores[:50]])
        late_avg = np.mean([s.score for s in all_scores[-50:]])

        # Late scores should be higher than early scores
        assert late_avg > early_avg

        # Should have some valid scores
        assert all(s.score >= 0 for s in all_scores)
        assert all(s.score <= 100 for s in all_scores)

        # All scores should have valid fields
        for s in all_scores[-10:]:
            d = s.to_dict()
            assert "player_id" in d
            assert "level" in d
            assert d["level"] in ("low", "moderate", "high", "critical")
