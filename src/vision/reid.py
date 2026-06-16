"""Player re-identification — team classification + jersey OCR + appearance matching.

Identity resolution strategy (layered):
1. Team color classification — constrains matching to same-team only
2. Jersey number (from OCR) — strongest identity signal across camera cuts
3. Color histogram similarity (team-constrained) — fallback for unknown jerseys
4. Post-game track merging — consolidates remaining fragments

The ReID module maps volatile track IDs to stable player identities.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .jersey import JerseyDetector
from .team_classifier import TeamClassifier

logger = logging.getLogger(__name__)


@dataclass
class PlayerIdentity:
    """Persistent player identity across camera cuts."""

    player_id: int
    jersey_number: int | None = None
    team_id: int = -1  # 0, 1, or -1 (unknown/referee)
    embeddings: list[np.ndarray] = field(default_factory=list)
    color_histogram: np.ndarray | None = None
    last_seen_frame: int = 0
    track_ids: list[int] = field(default_factory=list)

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings[-10:], axis=0)

    def add_embedding(self, embedding: np.ndarray, max_gallery: int = 10) -> None:
        self.embeddings.append(embedding)
        if len(self.embeddings) > max_gallery:
            self.embeddings = self.embeddings[-max_gallery:]

    def to_dict(self) -> dict:
        """Serialize for track merger."""
        return {
            "player_id": self.player_id,
            "jersey_number": self.jersey_number,
            "team": self.team_id,
            "color_histogram": self.color_histogram.tolist() if self.color_histogram is not None else None,
            "track_ids": self.track_ids,
        }


class PlayerReID:
    """Re-identification engine — team classification + jersey numbers + appearance.

    Team classification auto-calibrates from the first 60 player crops, then
    constrains all appearance matching to same-team only. This eliminates
    cross-team mismatches and cuts the matching search space in half.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        match_threshold: float = 0.50,  # Lowered from 0.65 — team constraint makes this safer
        gallery_size: int = 10,
    ) -> None:
        self.device = device
        self.match_threshold = match_threshold
        self.gallery_size = gallery_size

        self._reid_model = None
        self._model_name = model_name
        self._jersey_detector = JerseyDetector(device=device)
        self._team_classifier = TeamClassifier(calibration_samples=60)

        # Identity state
        self._identities: dict[int, PlayerIdentity] = {}
        self._track_to_player: dict[int, int] = {}
        self._jersey_team_to_player: dict[tuple[int, int], int] = {}  # (jersey, team) → player_id
        self._next_player_id = 1
        self._camera_cut_detected = False

    @property
    def team_classifier(self) -> TeamClassifier:
        return self._team_classifier

    def load_model(self) -> None:
        """Load ReID and jersey detection models."""
        self._jersey_detector.load_model()

        try:
            from torchreid.utils import FeatureExtractor
            self._reid_model = FeatureExtractor(
                model_name=self._model_name,
                device=self.device,
            )
            logger.info("Loaded re-ID model: %s", self._model_name)
        except ImportError:
            logger.info("torchreid not available — using team classification + jersey OCR + color histograms")

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        if self._reid_model is None:
            return None
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            features = self._reid_model(rgb)
            return F.normalize(features, dim=1).cpu().numpy().flatten()
        except Exception as e:
            logger.debug("Embedding extraction failed: %s", e)
            return None

    def extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        jersey_region = crop[: int(h * 0.6), :]
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def detect_camera_cut(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> bool:
        if prev_frame is None:
            return False
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
        prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
        cv2.normalize(curr_hist, curr_hist)
        cv2.normalize(prev_hist, prev_hist)
        diff = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
        self._camera_cut_detected = diff > 0.5
        return self._camera_cut_detected

    def match_track(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_number: int,
    ) -> int:
        """Match a track to a player identity. Returns player_id.

        Matching pipeline:
        1. Classify team (constrains all subsequent matching)
        2. Try jersey number → (jersey, team) → player_id
        3. If no camera cut and track mapped → keep
        4. Appearance match (team-constrained)
        5. Create new identity
        """
        # --- Team classification ---
        team_id = self._team_classifier.classify(crop, track_id)

        # --- Try jersey number ---
        jersey = self._jersey_detector.detect(crop, track_id, frame_number)

        if jersey is not None and team_id >= 0:
            key = (jersey, team_id)
            if key in self._jersey_team_to_player:
                player_id = self._jersey_team_to_player[key]
                self._track_to_player[track_id] = player_id
                identity = self._identities[player_id]
                identity.last_seen_frame = frame_number
                identity.team_id = team_id
                if track_id not in identity.track_ids:
                    identity.track_ids.append(track_id)
                return player_id
            else:
                # New jersey+team combo
                if track_id in self._track_to_player:
                    player_id = self._track_to_player[track_id]
                    identity = self._identities[player_id]
                else:
                    player_id = self._next_player_id
                    self._next_player_id += 1
                    identity = PlayerIdentity(player_id=player_id, last_seen_frame=frame_number)
                    self._identities[player_id] = identity

                self._jersey_team_to_player[key] = player_id
                identity.jersey_number = jersey
                identity.team_id = team_id
                identity.color_histogram = self.extract_color_histogram(crop)
                self._track_to_player[track_id] = player_id
                if track_id not in identity.track_ids:
                    identity.track_ids.append(track_id)
                return player_id

        # --- No jersey — continuity check ---
        if track_id in self._track_to_player and not self._camera_cut_detected:
            player_id = self._track_to_player[track_id]
            identity = self._identities[player_id]
            identity.last_seen_frame = frame_number
            if team_id >= 0:
                identity.team_id = team_id
            if frame_number % 30 == 0:
                embedding = self.extract_embedding(crop)
                if embedding is not None:
                    identity.add_embedding(embedding, self.gallery_size)
                identity.color_histogram = self.extract_color_histogram(crop)
            return player_id

        # --- Appearance matching (team-constrained) ---
        embedding = self.extract_embedding(crop)
        color_hist = self.extract_color_histogram(crop)

        best_match_id: int | None = None
        best_score = 0.0

        for pid, identity in self._identities.items():
            # Team constraint: skip different-team identities
            if team_id >= 0 and identity.team_id >= 0 and team_id != identity.team_id:
                continue

            # Skip identities seen very recently (likely different player in same frame)
            if abs(identity.last_seen_frame - frame_number) < 3:
                continue

            score = self._compute_similarity(embedding, color_hist, identity)
            if score > best_score:
                best_score = score
                best_match_id = pid

        if best_match_id is not None and best_score >= self.match_threshold:
            identity = self._identities[best_match_id]
            if embedding is not None:
                identity.add_embedding(embedding, self.gallery_size)
            identity.color_histogram = color_hist
            identity.last_seen_frame = frame_number
            if team_id >= 0:
                identity.team_id = team_id
            identity.track_ids.append(track_id)
            self._track_to_player[track_id] = best_match_id
            return best_match_id

        # --- New identity ---
        player_id = self._next_player_id
        self._next_player_id += 1
        identity = PlayerIdentity(
            player_id=player_id,
            team_id=team_id,
            last_seen_frame=frame_number,
        )
        if embedding is not None:
            identity.add_embedding(embedding, self.gallery_size)
        identity.color_histogram = color_hist
        identity.track_ids.append(track_id)
        self._identities[player_id] = identity
        self._track_to_player[track_id] = player_id
        return player_id

    def _compute_similarity(
        self,
        embedding: np.ndarray | None,
        color_hist: np.ndarray,
        identity: PlayerIdentity,
    ) -> float:
        scores: list[float] = []

        if embedding is not None and identity.mean_embedding is not None:
            cos_sim = float(np.dot(embedding, identity.mean_embedding))
            scores.append(cos_sim * 0.7)

        if identity.color_histogram is not None:
            hist_sim = float(cv2.compareHist(
                color_hist.astype(np.float32),
                identity.color_histogram.astype(np.float32),
                cv2.HISTCMP_CORREL,
            ))
            weight = 0.3 if scores else 1.0
            scores.append(hist_sim * weight)

        return sum(scores) if scores else 0.0

    def get_identity(self, player_id: int) -> PlayerIdentity | None:
        return self._identities.get(player_id)

    def get_all_identities(self) -> dict[int, PlayerIdentity]:
        return dict(self._identities)

    def get_identities_for_merger(self) -> dict[int, dict]:
        """Export identity data for post-game track merging."""
        return {pid: identity.to_dict() for pid, identity in self._identities.items()}

    def reset(self) -> None:
        self._identities.clear()
        self._track_to_player.clear()
        self._jersey_team_to_player.clear()
        self._next_player_id = 1
        self._jersey_detector.reset()
        self._team_classifier.reset()
