"""Video annotator — draws bounding boxes, player IDs, and fatigue scores on frames.

Produces an annotated MP4 alongside the analysis JSON. Each player gets:
- Colored bounding box (green→yellow→orange→red based on fatigue level)
- Player ID label
- Fatigue score bar overlay
- Team color indicator dot
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .pipeline import FrameResult
from ..models.fatigue import FatigueScore

logger = logging.getLogger(__name__)

# Fatigue level → BGR color
LEVEL_COLORS = {
    "low": (0, 200, 0),        # Green
    "moderate": (0, 200, 255),  # Yellow
    "high": (0, 140, 255),     # Orange
    "critical": (0, 0, 255),   # Red
}

TEAM_COLORS = {
    0: (255, 160, 50),   # Blue-ish (team 0)
    1: (50, 50, 255),    # Red-ish (team 1)
    -1: (180, 180, 180), # Gray (referee/unknown)
}


class VideoAnnotator:
    """Draws detection boxes and fatigue data onto video frames and writes to MP4."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    def open(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        logger.info("Annotator writing to: %s (%dx%d @ %.0ffps)", self.output_path, self.width, self.height, self.fps)

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: FrameResult,
        fatigue_scores: dict[int, FatigueScore],
        team_ids: dict[int, int] | None = None,
    ) -> np.ndarray:
        """Draw annotations on a frame and write it.

        Args:
            frame: Original BGR frame
            result: Vision pipeline output for this frame
            fatigue_scores: Player fatigue scores
            team_ids: Optional player_id → team_id mapping

        Returns:
            The annotated frame
        """
        annotated = frame.copy()

        for track in result.tracks:
            tid = track.track_id
            pid = result.player_ids.get(tid)
            if pid is None:
                continue

            bbox = track.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Get fatigue info
            score = fatigue_scores.get(pid)
            fatigue_val = score.score if score else 0.0
            level = score.level if score else "low"

            # Box color based on fatigue
            color = LEVEL_COLORS.get(level, (200, 200, 200))

            # Draw bounding box
            thickness = 2 if level in ("low", "moderate") else 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Team dot
            team_id = team_ids.get(pid, -1) if team_ids else -1
            team_color = TEAM_COLORS.get(team_id, (180, 180, 180))
            cv2.circle(annotated, (x1 + 8, y1 - 8), 5, team_color, -1)
            cv2.circle(annotated, (x1 + 8, y1 - 8), 5, (40, 40, 40), 1)

            # Label background
            label = f"P{pid}"
            fatigue_text = f"{fatigue_val:.0f}"
            label_full = f"{label} | {fatigue_text}"
            (tw, th), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 4, th + 4)
            cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 2), (0, 0, 0), -1)
            cv2.putText(annotated, label_full, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # Fatigue bar under the box
            bar_w = x2 - x1
            bar_h = 4
            bar_y = y2 + 2
            if bar_y + bar_h < annotated.shape[0]:
                # Background
                cv2.rectangle(annotated, (x1, bar_y), (x2, bar_y + bar_h), (40, 40, 40), -1)
                # Fill
                fill_w = int(bar_w * min(fatigue_val / 100, 1.0))
                if fill_w > 0:
                    cv2.rectangle(annotated, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), color, -1)

        # HUD overlay — top-left game info
        self._draw_hud(annotated, result, fatigue_scores)

        # Write frame
        if self._writer:
            self._writer.write(annotated)
            self._frame_count += 1

        return annotated

    def _draw_hud(
        self,
        frame: np.ndarray,
        result: FrameResult,
        fatigue_scores: dict[int, FatigueScore],
    ) -> None:
        """Draw heads-up display in top-left corner."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (280, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Game time
        game_time_s = result.timestamp_ms / 1000
        minutes = int(game_time_s // 60)
        seconds = int(game_time_s % 60)
        cv2.putText(frame, f"{minutes:02d}:{seconds:02d}", (16, 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Player count + avg fatigue
        n_players = len(result.tracks)
        scores = [s.score for s in fatigue_scores.values() if s.score > 0]
        avg_fatigue = sum(scores) / len(scores) if scores else 0
        max_fatigue = max(scores) if scores else 0

        cv2.putText(frame, f"Players: {n_players}", (16, 58),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Avg fatigue: {avg_fatigue:.0f}  Max: {max_fatigue:.0f}", (16, 78),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Camera cut indicator
        if result.camera_cut:
            cv2.putText(frame, "CAMERA CUT", (w - 160, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    def close(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        logger.info("Annotated video: %d frames written to %s", self._frame_count, self.output_path)

    def __enter__(self) -> VideoAnnotator:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
