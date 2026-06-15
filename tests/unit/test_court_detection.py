"""Tests for court line detection and homography computation."""

import numpy as np
import pytest

from src.vision.court import CourtDetector


class TestCourtLineDetection:
    def test_detect_court_lines_returns_mask(self):
        detector = CourtDetector()
        # Create a synthetic frame with white lines on brown background
        frame = np.full((480, 640, 3), [60, 90, 140], dtype=np.uint8)  # Brown-ish
        # Draw a white horizontal line
        frame[240, 100:540, :] = [255, 255, 255]
        frame[241, 100:540, :] = [255, 255, 255]
        # Draw a white vertical line
        frame[100:380, 320, :] = [255, 255, 255]
        frame[100:380, 321, :] = [255, 255, 255]

        mask = detector._detect_court_lines(frame)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        # The white lines should appear in the mask
        assert mask[240, 320] > 0

    def test_find_lines_on_empty_mask(self):
        detector = CourtDetector()
        mask = np.zeros((480, 640), dtype=np.uint8)
        lines = detector._find_lines(mask)
        assert lines is None or len(lines) == 0

    def test_find_intersections_parallel(self):
        """Parallel lines should produce no intersections."""
        detector = CourtDetector()
        # Two horizontal parallel lines
        lines = np.array([
            [[10, 100, 600, 100]],
            [[10, 200, 600, 200]],
        ])
        intersections = detector._find_intersections(lines, (480, 640))
        assert len(intersections) == 0

    def test_find_intersections_perpendicular(self):
        """Perpendicular lines should produce one intersection."""
        detector = CourtDetector()
        lines = np.array([
            [[0, 240, 640, 240]],    # Horizontal through center
            [[320, 0, 320, 480]],    # Vertical through center
        ])
        intersections = detector._find_intersections(lines, (480, 640))
        assert len(intersections) == 1
        x, y = intersections[0]
        assert abs(x - 320) < 1
        assert abs(y - 240) < 1

    def test_line_intersection_calculation(self):
        detector = CourtDetector()
        # X pattern crossing at (50, 50)
        line1 = np.array([0, 0, 100, 100])
        line2 = np.array([0, 100, 100, 0])
        pt = detector._line_intersection(line1, line2)
        assert pt is not None
        assert abs(pt[0] - 50) < 1e-5
        assert abs(pt[1] - 50) < 1e-5

    def test_line_intersection_parallel_returns_none(self):
        detector = CourtDetector()
        line1 = np.array([0, 0, 100, 0])
        line2 = np.array([0, 10, 100, 10])
        assert detector._line_intersection(line1, line2) is None

    def test_match_landmarks_empty(self):
        detector = CourtDetector()
        pixel, court = detector._match_landmarks([], (480, 640))
        assert pixel == []
        assert court == []

    def test_match_landmarks_too_few_points(self):
        detector = CourtDetector()
        pixel, court = detector._match_landmarks(
            [(100, 100), (200, 200), (300, 300)],  # Only 3 points
            (480, 640),
        )
        assert pixel == []
        assert court == []

    def test_detect_and_map_uses_cached_when_not_due(self):
        detector = CourtDetector(update_interval=300)
        # Set a fake existing mapping
        from src.vision.court import CourtMapping
        fake_mapping = CourtMapping(
            homography=np.eye(3),
            inverse_homography=np.eye(3),
            confidence=0.95,
            frame_number=0,
        )
        detector._current_mapping = fake_mapping
        detector._last_update_frame = 0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_and_map(frame, frame_number=100)

        # Should return cached mapping since interval not reached
        assert result is fake_mapping
