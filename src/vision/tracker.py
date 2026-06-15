"""ByteTrack multi-object tracker — maintains player identity within continuous footage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A tracked player across frames."""

    track_id: int
    bbox: np.ndarray              # Current [x1, y1, x2, y2]
    confidence: float
    age: int = 0                  # Total frames since track was created
    hits: int = 1                 # Frames where detection was matched
    time_since_update: int = 0    # Frames since last matched detection
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed after enough consistent matches."""
        return self.hits >= 3

    def predict(self) -> np.ndarray:
        """Predict next position using constant velocity model."""
        predicted_center = self.center + self.velocity
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return np.array([
            predicted_center[0] - w / 2,
            predicted_center[1] - h / 2,
            predicted_center[0] + w / 2,
            predicted_center[1] + h / 2,
        ])

    def update(self, detection: Detection) -> None:
        """Update track with a matched detection."""
        old_center = self.center.copy()
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        new_center = self.center
        # Exponential moving average for velocity
        self.velocity = 0.7 * self.velocity + 0.3 * (new_center - old_center)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.history.append(self.bbox.copy())

    def mark_missed(self) -> None:
        """Mark track as not detected in this frame."""
        self.time_since_update += 1
        self.age += 1
        # Use predicted position
        self.bbox = self.predict()


def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of bboxes. Shape: (N, M)."""
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0].T)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


class ByteTracker:
    """ByteTrack-style multi-object tracker.

    Two-stage association: first match high-confidence detections to tracks,
    then match remaining low-confidence detections to unmatched tracks.
    This recovers occluded players that YOLO detects with low confidence.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 60,
        min_track_length: int = 10,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_track_length = min_track_length

        self._tracks: list[Track] = []
        self._lost_tracks: list[Track] = []
        self._next_id = 1

    @property
    def active_tracks(self) -> list[Track]:
        """Return confirmed, currently-tracked players."""
        return [t for t in self._tracks if t.is_confirmed]

    def update(self, detections: list[Detection]) -> list[Track]:
        """Process detections for one frame, return active tracks."""
        # Split detections by confidence
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [
            d for d in detections
            if self.low_thresh <= d.confidence < self.high_thresh
        ]

        # Predict new positions for all tracks (update bbox for IoU matching)
        for track in self._tracks:
            track.bbox = track.predict()

        # --- Stage 1: Match high-confidence detections to existing tracks ---
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            self._tracks, high_dets, self.match_thresh
        )

        for track, det in matched_tracks:
            track.update(det)

        # --- Stage 2: Match low-confidence detections to remaining tracks ---
        if low_dets and unmatched_tracks:
            matched_low, still_unmatched, _ = self._associate(
                unmatched_tracks, low_dets, 0.5
            )
            for track, det in matched_low:
                track.update(det)
            unmatched_tracks = still_unmatched

        # --- Stage 3: Try to recover lost tracks with unmatched detections ---
        if unmatched_dets and self._lost_tracks:
            matched_lost, _, remaining_dets = self._associate(
                self._lost_tracks, unmatched_dets, 0.6
            )
            for track, det in matched_lost:
                track.update(det)
                self._tracks.append(track)
                self._lost_tracks.remove(track)
            unmatched_dets = remaining_dets

        # Handle unmatched tracks — move to lost
        for track in unmatched_tracks:
            track.mark_missed()
            if track.time_since_update > self.track_buffer:
                self._tracks.remove(track)
            elif track.time_since_update > 5:
                self._tracks.remove(track)
                self._lost_tracks.append(track)

        # Create new tracks for unmatched high-confidence detections
        for det in unmatched_dets:
            track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                confidence=det.confidence,
            )
            self._next_id += 1
            self._tracks.append(track)

        # Clean up old lost tracks
        self._lost_tracks = [
            t for t in self._lost_tracks if t.time_since_update <= self.track_buffer
        ]

        return self.active_tracks

    def _associate(
        self,
        tracks: list[Track],
        detections: list[Detection],
        thresh: float,
    ) -> tuple[
        list[tuple[Track, Detection]],
        list[Track],
        list[Detection],
    ]:
        """Hungarian algorithm matching using IoU cost matrix."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = np.array([d.bbox for d in detections])
        iou_matrix = _iou_batch(track_bboxes, det_bboxes)

        # Cost matrix = 1 - IoU (minimize cost)
        cost = 1.0 - iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost)

        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= (1.0 - thresh):
                matched.append((tracks[r], detections[c]))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)

        return (
            matched,
            [tracks[i] for i in unmatched_tracks],
            [detections[i] for i in unmatched_dets],
        )
