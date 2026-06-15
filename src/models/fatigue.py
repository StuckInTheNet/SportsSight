"""Fatigue model — temporal transformer over biomechanical feature sequences."""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..features.extractor import FEATURE_DIM, PlayerFeatures

logger = logging.getLogger(__name__)


@dataclass
class FatigueScore:
    """Fatigue assessment for one player."""

    player_id: int
    timestamp_ms: float
    score: float               # 0 (fresh) to 100 (exhausted)
    confidence: float          # Model confidence in this score
    trend: str                 # "rising", "stable", "declining"
    baseline_deviation: float  # How far current features deviate from baseline
    contributing_factors: dict[str, float] = field(default_factory=dict)
    predicted_score_5min: float = 0.0  # Predicted fatigue in 5 minutes

    @property
    def level(self) -> str:
        if self.score < 30:
            return "low"
        if self.score < 55:
            return "moderate"
        if self.score < 75:
            return "high"
        return "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "timestamp_ms": self.timestamp_ms,
            "score": round(self.score, 1),
            "confidence": round(self.confidence, 2),
            "level": self.level,
            "trend": self.trend,
            "baseline_deviation": round(self.baseline_deviation, 3),
            "contributing_factors": {
                k: round(v, 3) for k, v in self.contributing_factors.items()
            },
            "predicted_score_5min": round(self.predicted_score_5min, 1),
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the transformer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FatigueTransformer(nn.Module):
    """Temporal transformer that processes a sequence of feature vectors
    and outputs a fatigue score + prediction.

    Architecture:
    - Input projection (FEATURE_DIM → d_model)
    - Positional encoding
    - Transformer encoder (self-attention over time steps)
    - Output heads: fatigue score, confidence, 5-min prediction
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.fatigue_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1], scaled to [0, 100]
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, feature_dim) — sequence of player features

        Returns:
            fatigue: (batch, 1) in [0, 1]
            confidence: (batch, 1) in [0, 1]
            prediction: (batch, 1) in [0, 1] — predicted fatigue 5 min ahead
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Use the last time step's representation
        last = x[:, -1, :]

        fatigue = self.fatigue_head(last)
        confidence = self.confidence_head(last)
        prediction = self.prediction_head(last)

        return fatigue, confidence, prediction


class FatigueModel:
    """High-level fatigue scoring model wrapping the transformer.

    Maintains per-player feature sequences and baseline statistics.
    Before training data is available, uses a rule-based scoring system
    based on feature deviation from first-quarter baselines.
    """

    def __init__(
        self,
        device: str = "cpu",
        baseline_window_minutes: float = 6.0,
        use_learned_model: bool = False,
    ) -> None:
        self.device = device
        self.baseline_window_ms = baseline_window_minutes * 60 * 1000
        self.use_learned_model = use_learned_model

        # Per-player state
        self._sequences: dict[int, deque] = defaultdict(lambda: deque(maxlen=600))
        self._baselines: dict[int, np.ndarray | None] = defaultdict(lambda: None)
        self._baseline_samples: dict[int, list] = defaultdict(list)
        self._prev_scores: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

        # Learned model (loaded when available)
        self._model: FatigueTransformer | None = None
        if use_learned_model:
            self._model = FatigueTransformer().to(device)
            self._model.eval()

    def load_weights(self, path: str) -> None:
        """Load trained model weights."""
        if self._model is None:
            self._model = FatigueTransformer().to(self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self.use_learned_model = True
        logger.info("Loaded fatigue model weights from %s", path)

    def update(self, features: dict[int, PlayerFeatures]) -> dict[int, FatigueScore]:
        """Process features for all players and return fatigue scores."""
        scores: dict[int, FatigueScore] = {}

        for pid, pf in features.items():
            feature_array = pf.to_array()
            self._sequences[pid].append((pf.timestamp_ms, feature_array))

            # Build baseline from early-game data
            if pf.timestamp_ms <= self.baseline_window_ms:
                self._baseline_samples[pid].append(feature_array)
                self._baselines[pid] = np.mean(self._baseline_samples[pid], axis=0)

            # Score the player
            if self.use_learned_model and self._model is not None:
                score = self._score_learned(pid, pf)
            else:
                score = self._score_rule_based(pid, pf)

            self._prev_scores[pid].append(score.score)
            scores[pid] = score

        return scores

    def _score_learned(self, pid: int, pf: PlayerFeatures) -> FatigueScore:
        """Score using the trained transformer model."""
        seq = list(self._sequences[pid])
        if len(seq) < 10:
            return self._score_rule_based(pid, pf)

        # Build input tensor
        features = np.array([s[1] for s in seq[-300:]])  # Last 300 time steps
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fatigue, confidence, prediction = self._model(tensor)

        score_val = float(fatigue[0, 0]) * 100
        conf_val = float(confidence[0, 0])
        pred_val = float(prediction[0, 0]) * 100

        return FatigueScore(
            player_id=pid,
            timestamp_ms=pf.timestamp_ms,
            score=score_val,
            confidence=conf_val,
            trend=self._compute_trend(pid),
            baseline_deviation=self._compute_deviation(pid, pf),
            predicted_score_5min=pred_val,
        )

    def _score_rule_based(self, pid: int, pf: PlayerFeatures) -> FatigueScore:
        """Rule-based fatigue scoring using deviation from baseline.

        This is the initial scoring system before we have training data.
        Each feature's deviation from baseline contributes to the score,
        weighted by its importance as a fatigue indicator.
        """
        baseline = self._baselines.get(pid)
        current = pf.to_array()

        if baseline is None:
            return FatigueScore(
                player_id=pid,
                timestamp_ms=pf.timestamp_ms,
                score=0.0,
                confidence=0.3,
                trend="stable",
                baseline_deviation=0.0,
            )

        # Feature weights — how much each metric matters for fatigue
        weights = np.array([
            0.15,  # speed — declining speed is a primary fatigue signal
            0.08,  # acceleration — slower acceleration = fatigue
            0.10,  # deceleration — inability to brake hard
            0.06,  # lateral_speed — defensive quickness
            0.12,  # max_speed_window — peak output declining
            0.05,  # stride_length — shortening strides
            0.05,  # stride_frequency — altered gait
            0.08,  # defensive_stance_depth — standing up more
            0.04,  # hip_drop — lower center of gravity
            0.10,  # jump_height — can't get as high
            0.03,  # contest_frequency — contesting fewer shots
            0.06,  # recovery_time — taking longer to recover
            0.02,  # sprint_count
            0.03,  # torso_lean — leaning forward more
            0.02,  # shoulder_asymmetry
            0.01,  # distance_traveled
        ], dtype=np.float32)

        # Compute per-feature deviation
        deviation = np.zeros_like(current)
        for i in range(len(current)):
            if baseline[i] > 1e-6:
                # Negative deviation = decline = fatigue
                deviation[i] = (baseline[i] - current[i]) / baseline[i]
            else:
                deviation[i] = 0.0

        # Weighted fatigue score
        raw_score = np.sum(np.clip(deviation, 0, 1) * weights) * 100
        fatigue_score = np.clip(raw_score, 0, 100)

        # Contributing factors
        factor_names = [
            "speed", "acceleration", "deceleration", "lateral_speed",
            "max_speed", "stride_length", "stride_freq", "stance_depth",
            "hip_drop", "jump_height", "contest_freq", "recovery_time",
            "sprint_count", "torso_lean", "shoulder_asym", "distance",
        ]
        contributing = {
            name: float(dev * w)
            for name, dev, w in zip(factor_names, deviation, weights)
            if dev * w > 0.01
        }

        # Simple 5-min prediction: extrapolate current trend
        trend = self._compute_trend(pid)
        prev = list(self._prev_scores[pid])
        if len(prev) >= 5:
            rate = (prev[-1] - prev[-5]) / 5
            predicted = fatigue_score + rate * 75  # 75 more steps ≈ 5 min at 15fps
        else:
            predicted = fatigue_score

        return FatigueScore(
            player_id=pid,
            timestamp_ms=pf.timestamp_ms,
            score=float(fatigue_score),
            confidence=min(0.9, 0.3 + len(self._baseline_samples.get(pid, [])) * 0.01),
            trend=trend,
            baseline_deviation=float(np.mean(np.abs(deviation))),
            contributing_factors=contributing,
            predicted_score_5min=float(np.clip(predicted, 0, 100)),
        )

    def _compute_trend(self, pid: int) -> str:
        """Determine if fatigue is rising, stable, or declining."""
        prev = list(self._prev_scores.get(pid, []))
        if len(prev) < 5:
            return "stable"
        recent = np.mean(prev[-5:])
        older = np.mean(prev[-15:-5]) if len(prev) >= 15 else np.mean(prev[:5])
        diff = recent - older
        if diff > 3:
            return "rising"
        if diff < -3:
            return "declining"
        return "stable"

    def _compute_deviation(self, pid: int, pf: PlayerFeatures) -> float:
        baseline = self._baselines.get(pid)
        if baseline is None:
            return 0.0
        current = pf.to_array()
        return float(np.mean(np.abs(current - baseline) / (np.abs(baseline) + 1e-6)))
