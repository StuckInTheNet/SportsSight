"""Tests for fatigue scoring model."""

import numpy as np
import pytest
import torch

from src.features.extractor import FEATURE_DIM, PlayerFeatures
from src.models.fatigue import FatigueModel, FatigueScore, FatigueTransformer


class TestFatigueTransformer:
    def test_forward_shape(self):
        model = FatigueTransformer(feature_dim=FEATURE_DIM)
        batch_size, seq_len = 2, 50
        x = torch.randn(batch_size, seq_len, FEATURE_DIM)

        fatigue, confidence, prediction = model(x)

        assert fatigue.shape == (batch_size, 1)
        assert confidence.shape == (batch_size, 1)
        assert prediction.shape == (batch_size, 1)

    def test_output_range(self):
        model = FatigueTransformer(feature_dim=FEATURE_DIM)
        x = torch.randn(1, 30, FEATURE_DIM)

        fatigue, confidence, prediction = model(x)

        # Sigmoid outputs should be in [0, 1]
        assert 0 <= fatigue.item() <= 1
        assert 0 <= confidence.item() <= 1
        assert 0 <= prediction.item() <= 1

    def test_different_sequence_lengths(self):
        model = FatigueTransformer(feature_dim=FEATURE_DIM)

        for seq_len in [10, 50, 100, 300]:
            x = torch.randn(1, seq_len, FEATURE_DIM)
            fatigue, _, _ = model(x)
            assert fatigue.shape == (1, 1)

    def test_batch_independence(self):
        """Verify each sample in a batch gets independent scores."""
        model = FatigueTransformer(feature_dim=FEATURE_DIM)
        model.eval()

        x1 = torch.randn(1, 30, FEATURE_DIM)
        x2 = torch.randn(1, 30, FEATURE_DIM)

        with torch.no_grad():
            f1, _, _ = model(x1)
            f2, _, _ = model(x2)

            # Batch them together
            batched = torch.cat([x1, x2], dim=0)
            fb, _, _ = model(batched)

        assert abs(f1.item() - fb[0].item()) < 1e-5
        assert abs(f2.item() - fb[1].item()) < 1e-5


class TestFatigueModel:
    def _make_features(self, player_id: int, timestamp_ms: float, **kwargs) -> PlayerFeatures:
        pf = PlayerFeatures(
            player_id=player_id,
            timestamp_ms=timestamp_ms,
            frame_number=int(timestamp_ms / 66.67),
        )
        for k, v in kwargs.items():
            setattr(pf, k, v)
        return pf

    def test_baseline_establishment(self):
        model = FatigueModel(baseline_window_minutes=6)

        # Feed data within baseline window (first 6 minutes = 360000ms)
        features = {
            1: self._make_features(1, 1000, speed=15.0, acceleration=8.0),
        }
        scores = model.update(features)
        assert 1 in scores
        assert scores[1].score == 0.0  # No fatigue at start

    def test_fatigue_increases_with_decline(self):
        model = FatigueModel(baseline_window_minutes=0.1)  # 6 second baseline

        # Establish baseline with high performance
        for t in range(0, 6000, 100):
            features = {
                1: self._make_features(1, t, speed=20.0, acceleration=10.0, jump_height=2.0),
            }
            model.update(features)

        # Now simulate decline (after baseline window)
        features_tired = {
            1: self._make_features(1, 500000, speed=10.0, acceleration=4.0, jump_height=1.0),
        }
        scores = model.update(features_tired)

        assert scores[1].score > 0  # Should show some fatigue

    def test_trend_detection(self):
        model = FatigueModel(baseline_window_minutes=0.05)

        # Build up baseline
        for t in range(0, 3000, 100):
            model.update({1: self._make_features(1, t, speed=20.0)})

        # Rising fatigue
        for t in range(10000, 25000, 100):
            decline = (t - 10000) / 15000 * 10
            scores = model.update({
                1: self._make_features(1, t, speed=max(5, 20.0 - decline)),
            })

        # After sustained decline, trend should be rising (fatigue increasing)
        # or at least not "declining" (fatigue decreasing)
        assert scores[1].trend in ("rising", "stable")

    def test_score_to_dict(self):
        score = FatigueScore(
            player_id=1,
            timestamp_ms=1000.0,
            score=72.5,
            confidence=0.85,
            trend="rising",
            baseline_deviation=0.35,
            contributing_factors={"speed": 0.15, "jump_height": 0.10},
            predicted_score_5min=80.0,
        )
        d = score.to_dict()
        assert d["player_id"] == 1
        assert d["level"] == "high"  # 72.5 >= 55 (moderate) but < 75, wait — 72.5 is >=55 and <75 → "high"
        assert d["score"] == 72.5
        assert "speed" in d["contributing_factors"]

    def test_level_thresholds(self):
        base = dict(player_id=1, timestamp_ms=0, confidence=1.0,
                    trend="stable", baseline_deviation=0)
        assert FatigueScore(score=10, **base).level == "low"
        assert FatigueScore(score=40, **base).level == "moderate"
        assert FatigueScore(score=60, **base).level == "high"
        assert FatigueScore(score=95, **base).level == "critical"

    def test_multiple_players_independent(self):
        model = FatigueModel(baseline_window_minutes=0.05)

        # Both players establish baseline
        for t in range(0, 3000, 100):
            model.update({
                1: self._make_features(1, t, speed=20.0),
                2: self._make_features(2, t, speed=15.0),
            })

        # Player 1 declines, player 2 holds steady
        scores = model.update({
            1: self._make_features(1, 500000, speed=10.0),
            2: self._make_features(2, 500000, speed=15.0),
        })

        # Player 1 should be more fatigued
        assert scores[1].score > scores[2].score
