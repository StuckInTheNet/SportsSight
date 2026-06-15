"""Court homography — map pixel coordinates to real-world NBA court coordinates."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# NBA court dimensions in feet
NBA_COURT_LENGTH = 94.0
NBA_COURT_WIDTH = 50.0

# Known court reference points (feet from bottom-left corner)
# These are standard NBA markings used for homography calibration
NBA_REFERENCE_POINTS = {
    "center_court": (47.0, 25.0),
    "left_ft_line_left": (19.0, 17.0),
    "left_ft_line_right": (19.0, 33.0),
    "right_ft_line_left": (75.0, 17.0),
    "right_ft_line_right": (75.0, 33.0),
    "left_three_top": (14.0, 3.0),
    "left_three_bottom": (14.0, 47.0),
    "right_three_top": (80.0, 3.0),
    "right_three_bottom": (80.0, 47.0),
    "left_baseline_left": (0.0, 0.0),
    "left_baseline_right": (0.0, 50.0),
    "right_baseline_left": (94.0, 0.0),
    "right_baseline_right": (94.0, 50.0),
    "half_court_top": (47.0, 0.0),
    "half_court_bottom": (47.0, 50.0),
}


@dataclass
class CourtMapping:
    """Stores the homography matrix and provides coordinate transforms."""

    homography: np.ndarray         # 3x3 perspective transform matrix
    inverse_homography: np.ndarray # For mapping court coords back to pixel
    confidence: float              # How good the calibration is (0-1)
    frame_number: int              # Frame when this was computed

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinate to court position in feet."""
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

    def court_to_pixel(self, cx: float, cy: float) -> tuple[float, float]:
        """Convert court position (feet) to pixel coordinate."""
        point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography)
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

    def pixel_distance_to_feet(self, px1: float, py1: float, px2: float, py2: float) -> float:
        """Compute real-world distance in feet between two pixel positions."""
        c1 = self.pixel_to_court(px1, py1)
        c2 = self.pixel_to_court(px2, py2)
        return float(np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2))


class CourtDetector:
    """Detect NBA court lines and compute the homography matrix.

    Approach:
    1. Edge detection (Canny) to find lines
    2. Hough line transform to extract line segments
    3. Filter for court-colored lines (white/light on wood floor)
    4. Identify intersections that correspond to known court landmarks
    5. Compute homography from matched pixel↔court point pairs
    """

    def __init__(
        self,
        update_interval: int = 300,
        min_points: int = 4,
    ) -> None:
        self.update_interval = update_interval
        self.min_points = min_points
        self._current_mapping: CourtMapping | None = None
        self._last_update_frame: int = -999

    @property
    def mapping(self) -> CourtMapping | None:
        return self._current_mapping

    def should_update(self, frame_number: int) -> bool:
        return (frame_number - self._last_update_frame) >= self.update_interval

    def detect_and_map(self, frame: np.ndarray, frame_number: int) -> CourtMapping | None:
        """Detect court lines and compute homography for this frame.

        Returns CourtMapping if successful, None if court lines couldn't be found.
        """
        if not self.should_update(frame_number) and self._current_mapping:
            return self._current_mapping

        # Step 1: Isolate court lines (white on wood-colored background)
        court_mask = self._detect_court_lines(frame)

        # Step 2: Find line segments
        lines = self._find_lines(court_mask)
        if lines is None or len(lines) < self.min_points:
            logger.debug("Not enough court lines detected (%s)", len(lines) if lines is not None else 0)
            return self._current_mapping

        # Step 3: Find intersections → candidate reference points
        intersections = self._find_intersections(lines, frame.shape)

        # Step 4: Match intersections to known court landmarks
        pixel_points, court_points = self._match_landmarks(intersections, frame.shape)

        if len(pixel_points) < self.min_points:
            logger.debug("Not enough landmark matches: %d", len(pixel_points))
            return self._current_mapping

        # Step 5: Compute homography
        pixel_pts = np.array(pixel_points, dtype=np.float32)
        court_pts = np.array(court_points, dtype=np.float32)

        H, mask = cv2.findHomography(pixel_pts, court_pts, cv2.RANSAC, 5.0)
        if H is None:
            logger.debug("Homography computation failed")
            return self._current_mapping

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logger.debug("Homography matrix is singular — skipping update")
            return self._current_mapping
        inlier_ratio = float(np.sum(mask)) / len(mask) if mask is not None else 0.0

        self._current_mapping = CourtMapping(
            homography=H,
            inverse_homography=H_inv,
            confidence=inlier_ratio,
            frame_number=frame_number,
        )
        self._last_update_frame = frame_number
        logger.info(
            "Court homography updated at frame %d (confidence: %.2f, %d points)",
            frame_number, inlier_ratio, len(pixel_points),
        )
        return self._current_mapping

    def _detect_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """Isolate white court lines from the wood floor."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White lines: high value, low saturation
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _find_lines(self, mask: np.ndarray) -> np.ndarray | None:
        """Detect line segments using Hough transform."""
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=20,
        )
        return lines

    def _find_intersections(
        self, lines: np.ndarray, frame_shape: tuple[int, ...]
    ) -> list[tuple[float, float]]:
        """Find intersections between detected line segments."""
        h, w = frame_shape[:2]
        intersections: list[tuple[float, float]] = []

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._line_intersection(lines[i][0], lines[j][0])
                if pt is not None:
                    x, y = pt
                    # Filter to points within frame bounds (with margin)
                    if -w * 0.1 <= x <= w * 1.1 and -h * 0.1 <= y <= h * 1.1:
                        intersections.append((x, y))

        return intersections

    def _line_intersection(
        self, line1: np.ndarray, line2: np.ndarray
    ) -> tuple[float, float] | None:
        """Compute intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # Parallel lines

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    def _match_landmarks(
        self,
        intersections: list[tuple[float, float]],
        frame_shape: tuple[int, ...],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Match detected intersections to known court landmarks.

        Uses geometric heuristics based on the expected layout of an NBA court
        in a typical broadcast camera view.
        """
        h, w = frame_shape[:2]
        if not intersections:
            return [], []

        pixel_points: list[tuple[float, float]] = []
        court_points: list[tuple[float, float]] = []

        # Cluster intersections by spatial proximity
        pts = np.array(intersections)

        # Sort by x-coordinate to identify left/right sides
        sorted_by_x = pts[pts[:, 0].argsort()]

        # Heuristic: leftmost cluster = left side of court, rightmost = right
        n = len(sorted_by_x)
        if n < 4:
            return [], []

        # Take corner-like points and center-like points
        # Bottom-left region
        left_pts = sorted_by_x[: n // 3]
        right_pts = sorted_by_x[2 * n // 3 :]
        center_pts = sorted_by_x[n // 3 : 2 * n // 3]

        # Map extremes to court corners / center
        if len(left_pts) >= 2:
            top_left = left_pts[left_pts[:, 1].argmin()]
            bot_left = left_pts[left_pts[:, 1].argmax()]
            pixel_points.extend([(top_left[0], top_left[1]), (bot_left[0], bot_left[1])])
            court_points.extend([(0.0, 0.0), (0.0, 50.0)])

        if len(right_pts) >= 2:
            top_right = right_pts[right_pts[:, 1].argmin()]
            bot_right = right_pts[right_pts[:, 1].argmax()]
            pixel_points.extend([(top_right[0], top_right[1]), (bot_right[0], bot_right[1])])
            court_points.extend([(94.0, 0.0), (94.0, 50.0)])

        if len(center_pts) >= 1:
            # Center court point
            center = center_pts[np.argmin(np.abs(center_pts[:, 0] - w / 2))]
            pixel_points.append((center[0], center[1]))
            court_points.append((47.0, 25.0))

        return pixel_points, court_points
