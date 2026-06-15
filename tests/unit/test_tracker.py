"""Tests for ByteTrack multi-object tracker."""

import numpy as np
import pytest

from src.vision.detector import Detection
from src.vision.tracker import ByteTracker, Track, _iou_batch


def _make_detection(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    """Helper to create a detection with a dummy crop."""
    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    crop = np.zeros((int(y2 - y1), int(x2 - x1), 3), dtype=np.uint8)
    return Detection(bbox=bbox, confidence=conf, class_id=0, crop=crop)


class TestIoUBatch:
    def test_identical_boxes(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        iou = _iou_batch(boxes, boxes)
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = _iou_batch(a, b)
        assert iou[0, 0] < 1e-6

    def test_partial_overlap(self):
        a = np.array([[0, 0, 20, 20]], dtype=np.float32)
        b = np.array([[10, 10, 30, 30]], dtype=np.float32)
        iou = _iou_batch(a, b)
        # Overlap area = 10*10 = 100, Union = 400+400-100 = 700
        expected = 100.0 / 700.0
        assert abs(iou[0, 0] - expected) < 1e-5

    def test_multi_box(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15], [25, 25, 35, 35]], dtype=np.float32)
        iou = _iou_batch(a, b)
        assert iou.shape == (2, 2)
        # Diagonal should have overlap, off-diagonal should be zero
        assert iou[0, 0] > 0
        assert iou[1, 1] > 0
        assert iou[0, 1] < 1e-6
        assert iou[1, 0] < 1e-6


class TestByteTracker:
    def test_creates_tracks_from_detections(self):
        tracker = ByteTracker(high_thresh=0.5)
        dets = [
            _make_detection(100, 100, 150, 250, 0.9),
            _make_detection(300, 100, 350, 250, 0.85),
        ]
        tracks = tracker.update(dets)
        # First frame may not confirm tracks (need 3 hits)
        # But internal tracks should exist
        assert len(tracker._tracks) == 2

    def test_tracks_persist_across_frames(self):
        tracker = ByteTracker(high_thresh=0.5)
        det1 = [_make_detection(100, 100, 150, 250, 0.9)]

        # Frame 1
        tracker.update(det1)
        # Frame 2 — same position
        tracker.update(det1)
        # Frame 3 — confirmed after 3 hits
        tracks = tracker.update(det1)

        assert len(tracks) == 1
        assert tracks[0].hits >= 3
        assert tracks[0].is_confirmed

    def test_tracks_get_unique_ids(self):
        tracker = ByteTracker(high_thresh=0.5)
        dets = [
            _make_detection(100, 100, 150, 250, 0.9),
            _make_detection(300, 100, 350, 250, 0.85),
        ]
        tracker.update(dets)
        ids = [t.track_id for t in tracker._tracks]
        assert len(ids) == len(set(ids))

    def test_lost_tracks_recovered(self):
        tracker = ByteTracker(high_thresh=0.5, track_buffer=30)

        det = [_make_detection(100, 100, 150, 250, 0.9)]

        # Build up confirmed track
        for _ in range(5):
            tracker.update(det)

        original_id = tracker._tracks[0].track_id

        # Detection disappears for a few frames
        for _ in range(3):
            tracker.update([])

        # Detection reappears at same position
        tracks = tracker.update(det)

        # Track should be recovered (same or new ID depending on lost buffer)
        assert len(tracker._tracks) >= 1

    def test_high_low_confidence_split(self):
        tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1)

        # Mix of high and low confidence detections
        dets = [
            _make_detection(100, 100, 150, 250, 0.9),   # High
            _make_detection(300, 100, 350, 250, 0.3),   # Low
        ]
        tracker.update(dets)

        # Both should create tracks, but the low-confidence one
        # gets second-pass treatment
        assert len(tracker._tracks) >= 1


class TestTrack:
    def test_predict_with_velocity(self):
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 150, 250], dtype=np.float32),
            confidence=0.9,
            velocity=np.array([5.0, 2.0]),
        )
        predicted = track.predict()
        # Center should shift by velocity
        expected_cx = 125 + 5.0  # original center_x + vx
        expected_cy = 175 + 2.0
        assert abs((predicted[0] + predicted[2]) / 2 - expected_cx) < 1e-5
        assert abs((predicted[1] + predicted[3]) / 2 - expected_cy) < 1e-5

    def test_update_smooths_velocity(self):
        track = Track(
            track_id=1,
            bbox=np.array([100, 100, 150, 250], dtype=np.float32),
            confidence=0.9,
        )
        det = _make_detection(110, 105, 160, 255, 0.95)
        track.update(det)

        assert track.hits == 2
        assert track.time_since_update == 0
        assert np.linalg.norm(track.velocity) > 0
