"""Tests for biomechanical feature extraction."""

import numpy as np
import pytest

from src.features.extractor import FEATURE_DIM, FeatureExtractor, PlayerFeatures
from src.vision.pose import Skeleton


class TestPlayerFeatures:
    def test_to_array_length(self):
        pf = PlayerFeatures(player_id=1, timestamp_ms=0, frame_number=0, speed=10.0)
        arr = pf.to_array()
        assert arr.shape == (FEATURE_DIM,)
        assert arr.dtype == np.float32

    def test_to_array_matches_feature_dim(self):
        """FEATURE_DIM constant must match actual array size."""
        pf = PlayerFeatures(player_id=1, timestamp_ms=0, frame_number=0)
        assert len(pf.to_array()) == FEATURE_DIM

    def test_to_dict_completeness(self):
        pf = PlayerFeatures(
            player_id=1, timestamp_ms=1000, frame_number=15,
            speed=12.5, court_x=47.0, court_y=25.0,
        )
        d = pf.to_dict()
        assert d["player_id"] == 1
        assert d["speed"] == 12.5
        assert d["court_x"] == 47.0

    def test_default_values(self):
        pf = PlayerFeatures(player_id=1, timestamp_ms=0, frame_number=0)
        assert pf.speed == 0.0
        assert pf.jump_height == 0.0
        assert pf.sprint_count == 0


class TestSkeleton:
    def _make_skeleton(self) -> Skeleton:
        """Create a skeleton with reasonable keypoint positions."""
        # Standing player roughly centered
        keypoints = np.array([
            [100, 50],   # nose
            [95, 45],    # left_eye
            [105, 45],   # right_eye
            [90, 50],    # left_ear
            [110, 50],   # right_ear
            [80, 100],   # left_shoulder
            [120, 100],  # right_shoulder
            [70, 150],   # left_elbow
            [130, 150],  # right_elbow
            [65, 200],   # left_wrist
            [135, 200],  # right_wrist
            [85, 200],   # left_hip
            [115, 200],  # right_hip
            [80, 280],   # left_knee
            [120, 280],  # right_knee
            [80, 360],   # left_ankle
            [120, 360],  # right_ankle
        ], dtype=np.float32)
        confidences = np.ones(17, dtype=np.float32) * 0.9
        bbox = np.array([50, 30, 150, 380], dtype=np.float32)
        return Skeleton(keypoints=keypoints, confidences=confidences, bbox=bbox, player_id=1)

    def test_get_keypoint(self):
        skel = self._make_skeleton()
        x, y, c = skel.get_keypoint("nose")
        assert x == 100
        assert y == 50
        assert c == 0.9

    def test_knee_angles(self):
        skel = self._make_skeleton()
        left = skel.knee_angle_left
        right = skel.knee_angle_right
        assert left is not None
        assert right is not None
        # Standing upright — knee angle should be roughly 180 degrees
        assert 150 < left < 200
        assert 150 < right < 200

    def test_torso_lean(self):
        skel = self._make_skeleton()
        lean = skel.torso_lean
        assert lean is not None
        # Upright posture should have small lean
        assert lean < 20

    def test_hip_center(self):
        skel = self._make_skeleton()
        hip = skel.hip_center
        assert hip is not None
        assert hip[0] == 100.0  # midpoint of left(85) and right(115)

    def test_angle_with_low_confidence(self):
        skel = self._make_skeleton()
        skel.confidences[13] = 0.1  # left_knee low confidence
        assert skel.knee_angle_left is None  # Should return None
        assert skel.knee_angle_right is not None  # Right still valid
