"""Player re-identification — jersey numbers + color histograms.

Identity resolution strategy (in priority order):
1. Jersey number (from OCR) — strongest signal, stable across camera cuts
2. OSNet appearance embedding (if torchreid available) — good for same-angle continuity
3. HSV color histogram of jersey region — fallback when above unavailable

The ReID module maps volatile track IDs (which reset on camera cuts) to
stable player identities that persist across the entire game.
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

logger = logging.getLogger(__name__)


@dataclass
class PlayerIdentity:
    """Persistent player identity across camera cuts."""

    player_id: int
    jersey_number: int | None = None
    team: str | None = None
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


class PlayerReID:
    """Re-identification engine — jersey numbers + appearance features.

    Jersey number OCR is the primary identity signal. When a track gets a
    confirmed jersey number, it's mapped to that player for the rest of the
    game. Color histograms handle the gap before OCR confirms a number.
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "cpu",
        match_threshold: float = 0.65,
        gallery_size: int = 10,
    ) -> None:
        self.device = device
        self.match_threshold = match_threshold
        self.gallery_size = gallery_size

        self._reid_model = None
        self._model_name = model_name
        self._jersey_detector = JerseyDetector(device=device)

        # Identity state
        self._identities: dict[int, PlayerIdentity] = {}
        self._track_to_player: dict[int, int] = {}
        self._jersey_to_player: dict[int, int] = {}  # jersey_number → player_id
        self._next_player_id = 1
        self._camera_cut_detected = False

    def load_model(self) -> None:
        """Load ReID and jersey detection models."""
        # Jersey OCR (primary)
        self._jersey_detector.load_model()

        # OSNet ReID (optional secondary)
        try:
            from torchreid.utils import FeatureExtractor
            self._reid_model = FeatureExtractor(
                model_name=self._model_name,
                device=self.device,
            )
            logger.info("Loaded re-ID model: %s", self._model_name)
        except ImportError:
            logger.info(
                "torchreid not available — using jersey numbers + color histograms"
            )

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract appearance embedding from a player crop."""
        if self._reid_model is None:
            return None
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            features = self._reid_model(rgb)
            embedding = F.normalize(features, dim=1).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            logger.debug("Embedding extraction failed: %s", e)
            return None

    def extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Extract HSV color histogram from jersey area (upper 60% of crop)."""
        h = crop.shape[0]
        jersey_region = crop[: int(h * 0.6), :]
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def detect_camera_cut(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> bool:
        """Detect abrupt scene change via histogram difference."""
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

        Priority:
        1. If track already has a confirmed jersey number → use that
        2. If track has a jersey read that matches a known player → merge
        3. If no camera cut and track already mapped → keep mapping
        4. Otherwise → appearance matching or new identity
        """
        # --- Try jersey number identification ---
        jersey = self._jersey_detector.detect(crop, track_id, frame_number)

        if jersey is not None:
            # Check if this jersey number is already mapped to a player
            if jersey in self._jersey_to_player:
                player_id = self._jersey_to_player[jersey]
                self._track_to_player[track_id] = player_id
                identity = self._identities[player_id]
                identity.last_seen_frame = frame_number
                if track_id not in identity.track_ids:
                    identity.track_ids.append(track_id)
                return player_id
            else:
                # New jersey number → assign to existing or new player
                if track_id in self._track_to_player:
                    player_id = self._track_to_player[track_id]
                else:
                    player_id = self._next_player_id
                    self._next_player_id += 1
                    self._identities[player_id] = PlayerIdentity(
                        player_id=player_id,
                        last_seen_frame=frame_number,
                    )

                self._jersey_to_player[jersey] = player_id
                self._identities[player_id].jersey_number = jersey
                self._track_to_player[track_id] = player_id
                if track_id not in self._identities[player_id].track_ids:
                    self._identities[player_id].track_ids.append(track_id)
                logger.debug(
                    "Jersey #%d → Player %d (track %d, frame %d)",
                    jersey, player_id, track_id, frame_number,
                )
                return player_id

        # --- No jersey number yet — fall back to appearance ---

        # If no camera cut and track is already mapped, keep it
        if track_id in self._track_to_player and not self._camera_cut_detected:
            player_id = self._track_to_player[track_id]
            identity = self._identities[player_id]
            identity.last_seen_frame = frame_number
            # Periodically refresh appearance features
            if frame_number % 30 == 0:
                embedding = self.extract_embedding(crop)
                if embedding is not None:
                    identity.add_embedding(embedding, self.gallery_size)
                identity.color_histogram = self.extract_color_histogram(crop)
            return player_id

        # Need to match by appearance
        embedding = self.extract_embedding(crop)
        color_hist = self.extract_color_histogram(crop)

        best_match_id: int | None = None
        best_score = 0.0

        for pid, identity in self._identities.items():
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
            identity.track_ids.append(track_id)
            self._track_to_player[track_id] = best_match_id
            return best_match_id

        # New player
        player_id = self._next_player_id
        self._next_player_id += 1
        identity = PlayerIdentity(player_id=player_id, last_seen_frame=frame_number)
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
        """Combined similarity: embedding (70%) + color histogram (30%)."""
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

    def reset(self) -> None:
        """Clear all identity state (between games)."""
        self._identities.clear()
        self._track_to_player.clear()
        self._jersey_to_player.clear()
        self._next_player_id = 1
        self._jersey_detector.reset()
