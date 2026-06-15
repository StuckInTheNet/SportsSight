"""Tests for court homography module."""

import numpy as np
import pytest

from src.vision.court import CourtDetector, CourtMapping, NBA_COURT_LENGTH, NBA_COURT_WIDTH


class TestCourtMapping:
    def test_identity_transform(self):
        """With an identity homography, pixels map 1:1 to court coords."""
        H = np.eye(3, dtype=np.float64)
        mapping = CourtMapping(
            homography=H,
            inverse_homography=np.linalg.inv(H),
            confidence=1.0,
            frame_number=0,
        )

        cx, cy = mapping.pixel_to_court(47.0, 25.0)
        assert abs(cx - 47.0) < 1e-6
        assert abs(cy - 25.0) < 1e-6

    def test_roundtrip(self):
        """pixel → court → pixel should be identity."""
        # Create a simple scale transform (2x pixels per foot)
        H = np.array([
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        mapping = CourtMapping(
            homography=H,
            inverse_homography=np.linalg.inv(H),
            confidence=1.0,
            frame_number=0,
        )

        px, py = 200.0, 100.0
        cx, cy = mapping.pixel_to_court(px, py)
        px2, py2 = mapping.court_to_pixel(cx, cy)
        assert abs(px - px2) < 1e-3
        assert abs(py - py2) < 1e-3

    def test_distance_calculation(self):
        """Test real-world distance calculation."""
        H = np.eye(3, dtype=np.float64)
        mapping = CourtMapping(
            homography=H,
            inverse_homography=np.linalg.inv(H),
            confidence=1.0,
            frame_number=0,
        )

        # Pythagorean 3-4-5
        dist = mapping.pixel_distance_to_feet(0, 0, 3, 4)
        assert abs(dist - 5.0) < 1e-5


class TestCourtDetector:
    def test_should_update_initial(self):
        detector = CourtDetector(update_interval=300)
        assert detector.should_update(0) is True

    def test_should_update_respects_interval(self):
        detector = CourtDetector(update_interval=300)
        detector._last_update_frame = 100
        assert detector.should_update(200) is False
        assert detector.should_update(400) is True

    def test_constants(self):
        assert NBA_COURT_LENGTH == 94.0
        assert NBA_COURT_WIDTH == 50.0
