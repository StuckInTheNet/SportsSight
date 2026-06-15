"""Player re-identification — maintains identity across camera cuts and occlusions."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# Standard person re-ID preprocessing
REID_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class PlayerIdentity:
    """Persistent player identity across camera cuts."""

    player_id: int
    jersey_number: int | None = None
    team: str | None = None
    embeddings: list[np.ndarray] = field(default_factory=list)
    color_histogram: np.ndarray | None = None
    last_seen_frame: int = 0
    track_ids: list[int] = field(default_factory=list)  # All track IDs mapped to this player

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings[-10:], axis=0)  # Average last N embeddings

    def add_embedding(self, embedding: np.ndarray, max_gallery: int = 10) -> None:
        """Add an embedding, keeping gallery bounded."""
        self.embeddings.append(embedding)
        if len(self.embeddings) > max_gallery:
            self.embeddings = self.embeddings[-max_gallery:]


class PlayerReID:
    """Re-identification engine using OSNet embeddings + color histograms.

    This module solves the camera-cut problem: when the broadcast switches angles,
    ByteTrack loses track IDs because there's no spatial continuity. ReID matches
    players across these discontinuities using appearance features.

    Strategy:
    1. Extract a feature embedding from each player crop using OSNet
    2. Maintain a gallery of embeddings per known player identity
    3. When a new track appears (or after a camera cut), match it to the gallery
       using cosine similarity
    4. Color histograms provide a fast secondary signal (jersey colors)
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

        self._model: torch.nn.Module | None = None
        self._model_name = model_name
        self._identities: dict[int, PlayerIdentity] = {}
        self._track_to_player: dict[int, int] = {}
        self._next_player_id = 1
        self._camera_cut_detected = False

    def load_model(self) -> None:
        """Load the re-ID model. Deferred to avoid import-time GPU allocation."""
        try:
            from torchreid.utils import FeatureExtractor
            self._model = FeatureExtractor(
                model_name=self._model_name,
                device=self.device,
            )
            logger.info("Loaded re-ID model: %s on %s", self._model_name, self.device)
        except ImportError:
            logger.warning(
                "torchreid not available — falling back to color-histogram-only re-ID. "
                "Install with: pip install torchreid"
            )
            self._model = None

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract appearance embedding from a player crop."""
        if self._model is None:
            return None

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        try:
            # Convert BGR to RGB for the model
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = REID_TRANSFORM(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self._model(tensor)

            embedding = F.normalize(features, dim=1).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            logger.debug("Embedding extraction failed: %s", e)
            return None

    def extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Extract HSV color histogram from jersey area (upper 60% of crop).

        We focus on the upper body because legs are often occluded and shorts
        may differ less between teams than jerseys.
        """
        h = crop.shape[0]
        jersey_region = crop[: int(h * 0.6), :]

        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def detect_camera_cut(self, frame: np.ndarray, prev_frame: np.ndarray | None) -> bool:
        """Detect abrupt scene change (camera cut) via histogram difference."""
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
        """Match a track to a known player identity (or create a new one).

        Returns the player_id assigned to this track.
        """
        # If this track is already mapped and we haven't had a camera cut, keep it
        if track_id in self._track_to_player and not self._camera_cut_detected:
            player_id = self._track_to_player[track_id]
            identity = self._identities[player_id]
            identity.last_seen_frame = frame_number

            # Refresh the embedding gallery periodically
            if frame_number % 30 == 0:
                embedding = self.extract_embedding(crop)
                if embedding is not None:
                    identity.add_embedding(embedding, self.gallery_size)
                identity.color_histogram = self.extract_color_histogram(crop)

            return player_id

        # Need to match — extract features
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
            # Matched to existing player
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
        """Compute combined similarity score between a crop and a known identity."""
        scores: list[float] = []

        # Embedding similarity (cosine)
        if embedding is not None and identity.mean_embedding is not None:
            cos_sim = float(np.dot(embedding, identity.mean_embedding))
            scores.append(cos_sim * 0.7)  # 70% weight to embedding

        # Color histogram similarity
        if identity.color_histogram is not None:
            hist_sim = float(cv2.compareHist(
                color_hist.astype(np.float32),
                identity.color_histogram.astype(np.float32),
                cv2.HISTCMP_CORREL,
            ))
            weight = 0.3 if scores else 1.0  # Full weight if no embedding
            scores.append(hist_sim * weight)

        return sum(scores) if scores else 0.0

    def get_identity(self, player_id: int) -> PlayerIdentity | None:
        return self._identities.get(player_id)

    def get_all_identities(self) -> dict[int, PlayerIdentity]:
        return dict(self._identities)

    def reset(self) -> None:
        """Clear all identity state (e.g., between games)."""
        self._identities.clear()
        self._track_to_player.clear()
        self._next_player_id = 1
