"""Biomechanical feature extraction from skeleton sequences and court positions."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..vision.court import CourtMapping
from ..vision.pose import Skeleton
from ..vision.pipeline import FrameResult

logger = logging.getLogger(__name__)

# Speed thresholds in feet/second for NBA players
SPRINT_THRESHOLD_FPS = 15.0   # ~10 mph — fast break / closeout
JOG_THRESHOLD_FPS = 7.0       # ~5 mph
WALK_THRESHOLD_FPS = 3.0      # ~2 mph


@dataclass
class PlayerFeatures:
    """Computed features for one player at a point in time."""

    player_id: int
    timestamp_ms: float
    frame_number: int

    # Movement (feet/second, computed via court homography)
    speed: float = 0.0
    acceleration: float = 0.0
    deceleration: float = 0.0
    lateral_speed: float = 0.0
    max_speed_window: float = 0.0       # Peak speed in current window

    # Stride mechanics
    stride_length: float = 0.0          # Feet between ankle positions
    stride_frequency: float = 0.0       # Strides per second

    # Defensive biomechanics
    defensive_stance_depth: float = 0.0  # Average knee flexion angle
    hip_drop: float = 0.0               # How much hips have dropped vs baseline

    # Explosiveness
    jump_height: float = 0.0            # Estimated from hip vertical displacement
    contest_frequency: float = 0.0      # Jumps per minute

    # Recovery
    recovery_time: float = 0.0          # Seconds to return to baseline after sprint
    sprint_count: int = 0               # Total sprints in window

    # Posture
    torso_lean: float = 0.0             # Degrees from vertical
    shoulder_asymmetry: float = 0.0     # Left/right shoulder height diff

    # Court position
    court_x: float = 0.0               # Feet from left baseline
    court_y: float = 0.0               # Feet from bottom sideline
    distance_traveled: float = 0.0      # Cumulative feet in window

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def to_array(self) -> np.ndarray:
        """Convert numeric features to a flat array for model input."""
        return np.array([
            self.speed, self.acceleration, self.deceleration, self.lateral_speed,
            self.max_speed_window, self.stride_length, self.stride_frequency,
            self.defensive_stance_depth, self.hip_drop, self.jump_height,
            self.contest_frequency, self.recovery_time, self.sprint_count,
            self.torso_lean, self.shoulder_asymmetry, self.distance_traveled,
        ], dtype=np.float32)


# Number of float features in to_array()
FEATURE_DIM = 16


@dataclass
class _PlayerState:
    """Internal state for tracking a player's feature history."""

    positions: deque = field(default_factory=lambda: deque(maxlen=900))  # (timestamp_ms, x_ft, y_ft)
    speeds: deque = field(default_factory=lambda: deque(maxlen=900))
    skeletons: deque = field(default_factory=lambda: deque(maxlen=300))
    hip_heights: deque = field(default_factory=lambda: deque(maxlen=300))
    sprint_timestamps: list = field(default_factory=list)
    last_sprint_end: float = 0.0
    ankle_positions: deque = field(default_factory=lambda: deque(maxlen=60))


class FeatureExtractor:
    """Extracts biomechanical features per player from vision pipeline output.

    Maintains a rolling buffer per player and computes features over configurable
    time windows. Features are calibrated to real-world units (feet, feet/sec)
    via the court homography.
    """

    def __init__(
        self,
        fps: float = 15.0,
        windows: list[int] | None = None,
    ) -> None:
        self.fps = fps
        self.frame_dt = 1.0 / fps
        self.windows = windows or [30, 120, 300]  # seconds
        self._players: dict[int, _PlayerState] = defaultdict(_PlayerState)

    def process(self, result: FrameResult) -> dict[int, PlayerFeatures]:
        """Extract features for all players in a frame result."""
        features: dict[int, PlayerFeatures] = {}
        court = result.court_mapping

        for skeleton in result.skeletons:
            pid = skeleton.player_id
            state = self._players[pid]

            # Update position history (court coordinates)
            court_pos = self._get_court_position(skeleton, court)
            if court_pos:
                state.positions.append((result.timestamp_ms, court_pos[0], court_pos[1]))

            # Update skeleton history
            state.skeletons.append((result.timestamp_ms, skeleton))

            # Update ankle tracking for stride computation
            self._update_ankle_tracking(skeleton, state)

            # Update hip height for jump detection
            hip = skeleton.hip_center
            if hip:
                state.hip_heights.append((result.timestamp_ms, hip[1]))

            # Compute features
            pf = PlayerFeatures(
                player_id=pid,
                timestamp_ms=result.timestamp_ms,
                frame_number=result.frame_number,
            )

            self._compute_speed(pf, state)
            self._compute_stride(pf, state)
            self._compute_defensive_stance(pf, state)
            self._compute_jumps(pf, state)
            self._compute_recovery(pf, state)
            self._compute_posture(pf, skeleton)
            self._compute_court_position(pf, court_pos)
            self._compute_distance(pf, state)

            features[pid] = pf

        return features

    def _get_court_position(
        self, skeleton: Skeleton, court: CourtMapping | None
    ) -> tuple[float, float] | None:
        """Get player's position in court coordinates."""
        hip = skeleton.hip_center
        if not hip or not court:
            return None
        try:
            return court.pixel_to_court(hip[0], hip[1])
        except Exception:
            return None

    def _compute_speed(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Compute instantaneous and windowed speed from position history."""
        if len(state.positions) < 2:
            return

        # Instantaneous speed (last two positions)
        t1, x1, y1 = state.positions[-2]
        t2, x2, y2 = state.positions[-1]
        dt = (t2 - t1) / 1000.0  # Convert ms to seconds
        if dt <= 0:
            return

        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        speed = dist / dt

        pf.speed = speed
        pf.lateral_speed = abs(dy) / dt  # Lateral = along baseline

        state.speeds.append((t2, speed))

        # Acceleration (change in speed)
        if len(state.speeds) >= 3:
            _, s_prev = state.speeds[-2]
            accel = (speed - s_prev) / dt
            if accel > 0:
                pf.acceleration = accel
            else:
                pf.deceleration = abs(accel)

        # Track sprints
        if speed >= SPRINT_THRESHOLD_FPS:
            if not state.sprint_timestamps or (t2 - state.sprint_timestamps[-1]) > 2000:
                state.sprint_timestamps.append(t2)
        elif speed < JOG_THRESHOLD_FPS and state.sprint_timestamps:
            state.last_sprint_end = t2

        pf.sprint_count = len(state.sprint_timestamps)

        # Max speed in last 30 seconds
        cutoff = t2 - 30000
        recent_speeds = [s for t, s in state.speeds if t >= cutoff]
        if recent_speeds:
            pf.max_speed_window = max(recent_speeds)

    def _compute_stride(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Compute stride length and frequency from ankle keypoint oscillations."""
        ankles = list(state.ankle_positions)
        if len(ankles) < 10:
            return

        # Stride length: average distance between successive ankle peaks
        positions = np.array([a[1] for a in ankles])  # x positions
        if len(positions) < 4:
            return

        # Simple peak detection on ankle x-position oscillation
        diffs = np.diff(positions)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]

        if len(sign_changes) >= 2:
            stride_distances = []
            for i in range(1, len(sign_changes)):
                d = abs(positions[sign_changes[i]] - positions[sign_changes[i - 1]])
                stride_distances.append(d)
            if stride_distances:
                pf.stride_length = np.mean(stride_distances)

            # Stride frequency
            time_span = (ankles[-1][0] - ankles[0][0]) / 1000.0
            if time_span > 0:
                pf.stride_frequency = len(sign_changes) / (2 * time_span)

    def _update_ankle_tracking(self, skeleton: Skeleton, state: _PlayerState) -> None:
        la = skeleton.get_keypoint("left_ankle")
        ra = skeleton.get_keypoint("right_ankle")
        if la[2] > 0.3:
            state.ankle_positions.append((0, la[0]))
        if ra[2] > 0.3:
            state.ankle_positions.append((0, ra[0]))

    def _compute_defensive_stance(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Measure defensive stance from knee flexion angles."""
        if not state.skeletons:
            return
        _, skeleton = state.skeletons[-1]

        angles = []
        left_knee = skeleton.knee_angle_left
        right_knee = skeleton.knee_angle_right
        if left_knee is not None:
            angles.append(left_knee)
        if right_knee is not None:
            angles.append(right_knee)

        if angles:
            pf.defensive_stance_depth = np.mean(angles)

    def _compute_jumps(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Detect jumps from hip height oscillation."""
        if len(state.hip_heights) < 10:
            return

        heights = np.array([h for _, h in state.hip_heights])
        baseline = np.median(heights)

        # Jump = hip height significantly above baseline (in pixel space, lower y = higher)
        jump_frames = np.sum(heights < baseline - 20)
        total_time = (state.hip_heights[-1][0] - state.hip_heights[0][0]) / 1000.0

        if total_time > 0:
            pf.contest_frequency = (jump_frames / self.fps) / total_time * 60.0

        # Current jump height (rough estimate)
        current = heights[-1]
        if current < baseline - 15:
            pf.jump_height = (baseline - current) * 0.05  # Rough pixel-to-feet

    def _compute_recovery(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Compute time to recover from sprint to baseline speed."""
        if not state.sprint_timestamps or state.last_sprint_end <= 0:
            return

        last_sprint = state.sprint_timestamps[-1]
        if state.last_sprint_end > last_sprint:
            pf.recovery_time = (state.last_sprint_end - last_sprint) / 1000.0

    def _compute_posture(self, pf: PlayerFeatures, skeleton: Skeleton) -> None:
        """Extract posture metrics from skeleton."""
        lean = skeleton.torso_lean
        if lean is not None:
            pf.torso_lean = lean

        # Shoulder asymmetry
        ls = skeleton.get_keypoint("left_shoulder")
        rs = skeleton.get_keypoint("right_shoulder")
        if ls[2] > 0.3 and rs[2] > 0.3:
            pf.shoulder_asymmetry = abs(ls[1] - rs[1])

    def _compute_court_position(
        self, pf: PlayerFeatures, court_pos: tuple[float, float] | None
    ) -> None:
        if court_pos:
            pf.court_x, pf.court_y = court_pos

    def _compute_distance(self, pf: PlayerFeatures, state: _PlayerState) -> None:
        """Cumulative distance traveled in the current window."""
        if len(state.positions) < 2:
            return

        cutoff = state.positions[-1][0] - 30000  # Last 30 seconds
        total = 0.0
        positions = list(state.positions)
        for i in range(1, len(positions)):
            if positions[i][0] >= cutoff:
                dx = positions[i][1] - positions[i - 1][1]
                dy = positions[i][2] - positions[i - 1][2]
                total += np.sqrt(dx**2 + dy**2)

        pf.distance_traveled = total
