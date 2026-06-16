"""Team color classification — assigns players to teams based on jersey color.

NBA games have two teams with distinct jersey colors. This module clusters
players into teams using K-means on jersey color features, then uses team
assignment as a constraint for ReID matching (a player can only match to
identities on the same team).

The classifier auto-calibrates from the first N player crops it sees,
requiring no manual team color input.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class TeamColorProfile:
    """Learned color profile for one team."""

    team_id: int  # 0 or 1
    mean_hue: float
    mean_saturation: float
    mean_value: float
    centroid: np.ndarray  # K-means centroid in feature space


class TeamClassifier:
    """Classifies players into two teams based on jersey dominant color.

    How it works:
    1. Collect jersey color features from the first `calibration_samples` crops
    2. Run K-means (k=2) to find the two team color clusters
    3. For subsequent crops, assign to nearest cluster

    Also detects referees (predominantly gray/black) and excludes them.
    """

    def __init__(self, calibration_samples: int = 60) -> None:
        self.calibration_samples = calibration_samples
        self._calibrated = False
        self._kmeans: KMeans | None = None
        self._samples: list[np.ndarray] = []
        self._profiles: list[TeamColorProfile] = []
        # Cache: track_id → team_id (0 or 1, or -1 for referee/unknown)
        self._track_team: dict[int, int] = {}
        # Vote accumulator for noisy single-frame classifications
        self._track_votes: dict[int, Counter] = {}

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def classify(self, crop: np.ndarray, track_id: int) -> int:
        """Classify a player crop into team 0, team 1, or -1 (referee/unknown).

        Returns cached result if this track has been stably classified.
        """
        # Return cached if stable
        if track_id in self._track_team:
            return self._track_team[track_id]

        features = self._extract_color_features(crop)
        if features is None:
            return -1

        if not self._calibrated:
            self._samples.append(features)
            if len(self._samples) >= self.calibration_samples:
                self._calibrate()
            return -1

        # Classify using K-means
        team_id = int(self._kmeans.predict(features.reshape(1, -1))[0])

        # Check if this looks like a referee (low saturation = gray/black)
        if features[1] < 40:  # Mean saturation < 40 → likely referee
            team_id = -1

        # Vote-based stabilization
        if track_id not in self._track_votes:
            self._track_votes[track_id] = Counter()
        self._track_votes[track_id][team_id] += 1

        # Lock in after 3 agreeing votes
        votes = self._track_votes[track_id]
        best_team, best_count = votes.most_common(1)[0]
        if best_count >= 3:
            self._track_team[track_id] = best_team
            return best_team

        return team_id

    def get_team(self, track_id: int) -> int | None:
        """Get the confirmed team for a track, or None if not yet classified."""
        return self._track_team.get(track_id)

    def _extract_color_features(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract color features from the jersey region of a player crop."""
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        h = crop.shape[0]
        # Jersey region: upper 20-55% of crop (skip head, skip shorts)
        y_start = int(h * 0.20)
        y_end = int(h * 0.55)
        jersey = crop[y_start:y_end, :]

        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

        # Features: mean H, S, V + dominant hue bin
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        # Dominant hue (mode of hue channel, quantized to 18 bins)
        hue_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        dominant_hue = float(np.argmax(hue_hist)) * 10  # Scale to 0-170

        # Saturation histogram peak
        sat_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        dominant_sat = float(np.argmax(sat_hist)) * 32

        return np.array([mean_h, mean_s, mean_v, dominant_hue, dominant_sat])

    def _calibrate(self) -> None:
        """Run K-means on collected samples to find two team clusters."""
        X = np.array(self._samples)

        # Filter out likely referees (low saturation) before clustering
        mask = X[:, 1] >= 30  # saturation >= 30
        X_players = X[mask]

        if len(X_players) < 10:
            logger.warning("Not enough non-referee samples for calibration (%d)", len(X_players))
            self._samples.clear()
            return

        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self._kmeans.fit(X_players)

        # Store team profiles
        for i, centroid in enumerate(self._kmeans.cluster_centers_):
            self._profiles.append(TeamColorProfile(
                team_id=i,
                mean_hue=centroid[0],
                mean_saturation=centroid[1],
                mean_value=centroid[2],
                centroid=centroid,
            ))

        self._calibrated = True
        self._samples.clear()

        logger.info(
            "Team classifier calibrated: Team 0 HSV=(%.0f,%.0f,%.0f), Team 1 HSV=(%.0f,%.0f,%.0f)",
            self._profiles[0].mean_hue, self._profiles[0].mean_saturation, self._profiles[0].mean_value,
            self._profiles[1].mean_hue, self._profiles[1].mean_saturation, self._profiles[1].mean_value,
        )

    def reset(self) -> None:
        self._calibrated = False
        self._kmeans = None
        self._samples.clear()
        self._profiles.clear()
        self._track_team.clear()
        self._track_votes.clear()
