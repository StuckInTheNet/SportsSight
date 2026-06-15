"""Pose estimation — skeleton keypoint extraction using YOLOv8-Pose.

YOLOv8-Pose detects persons AND extracts 17 COCO keypoints in a single
forward pass on MPS/CUDA. This replaces the previous MMPose dependency
which had build issues on Python 3.13 and Apple Silicon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO 17-keypoint skeleton definition
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # Legs
]

# Keypoint index shortcuts for biomechanical calculations
KPT = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


@dataclass
class Skeleton:
    """Extracted skeleton for one player in one frame."""

    keypoints: np.ndarray      # (17, 2) — x, y in frame coordinates
    confidences: np.ndarray    # (17,) — per-keypoint confidence
    bbox: np.ndarray           # [x1, y1, x2, y2] — detection bbox in frame coords
    player_id: int             # Mapped player identity

    def get_keypoint(self, name: str) -> tuple[float, float, float]:
        """Get (x, y, confidence) for a named keypoint."""
        idx = KPT[name]
        return (self.keypoints[idx, 0], self.keypoints[idx, 1], self.confidences[idx])

    def get_angle(self, a: str, b: str, c: str) -> float | None:
        """Compute angle at joint b formed by segments a→b and b→c.

        Returns angle in degrees, or None if any keypoint has low confidence.
        """
        threshold = 0.3
        ax, ay, ac = self.get_keypoint(a)
        bx, by, bc = self.get_keypoint(b)
        cx, cy, cc = self.get_keypoint(c)

        if ac < threshold or bc < threshold or cc < threshold:
            return None

        ba = np.array([ax - bx, ay - by])
        bc_vec = np.array([cx - bx, cy - by])

        cos_angle = np.dot(ba, bc_vec) / (np.linalg.norm(ba) * np.linalg.norm(bc_vec) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return float(angle)

    @property
    def knee_angle_left(self) -> float | None:
        """Left knee flexion angle."""
        return self.get_angle("left_hip", "left_knee", "left_ankle")

    @property
    def knee_angle_right(self) -> float | None:
        """Right knee flexion angle."""
        return self.get_angle("right_hip", "right_knee", "right_ankle")

    @property
    def torso_lean(self) -> float | None:
        """Forward torso lean in degrees from vertical."""
        ls = self.get_keypoint("left_shoulder")
        rs = self.get_keypoint("right_shoulder")
        lh = self.get_keypoint("left_hip")
        rh = self.get_keypoint("right_hip")

        if any(c < 0.3 for _, _, c in [ls, rs, lh, rh]):
            return None

        mid_shoulder = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        mid_hip = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

        # Angle from vertical (straight up = 0 degrees)
        spine = mid_shoulder - mid_hip
        vertical = np.array([0, -1])  # Up in image coords
        cos_angle = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    @property
    def hip_center(self) -> tuple[float, float] | None:
        """Center of hips — best proxy for player position on court."""
        lh = self.get_keypoint("left_hip")
        rh = self.get_keypoint("right_hip")
        if lh[2] < 0.3 or rh[2] < 0.3:
            return None
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)


class PoseEstimator:
    """Pose estimation using YOLOv8-Pose.

    Runs the pose model on player crops extracted from their bounding boxes.
    YOLOv8-Pose outputs COCO 17-keypoint skeletons with per-keypoint confidence.
    """

    def __init__(self, device: str = "cpu", model: str = "yolov8x-pose") -> None:
        self.device = device
        self._model_name = model
        self._model: YOLO | None = None

    def load_model(self) -> None:
        """Load YOLOv8-Pose model."""
        model_file = f"{self._model_name}.pt"
        logger.info("Loading pose model: %s on %s", model_file, self.device)
        self._model = YOLO(model_file)
        logger.info("Pose model loaded: %s", self._model_name)

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        player_ids: list[int],
    ) -> list[Skeleton]:
        """Estimate poses for detected players.

        Runs YOLOv8-Pose on each player crop, then maps keypoints back
        to full-frame coordinates.

        Args:
            frame: Full BGR image
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            player_ids: Corresponding player IDs

        Returns:
            List of Skeleton objects
        """
        if not bboxes:
            return []

        if self._model is None:
            return self._empty_skeletons(bboxes, player_ids)

        return self._estimate_yolo_pose(frame, bboxes, player_ids)

    def _estimate_yolo_pose(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        player_ids: list[int],
    ) -> list[Skeleton]:
        """Run YOLOv8-Pose on player crops and map keypoints to frame coords."""
        skeletons: list[Skeleton] = []

        for bbox, pid in zip(bboxes, player_ids):
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                skeletons.append(self._empty_skeleton(bbox, pid))
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 32 or crop.shape[1] < 16:
                skeletons.append(self._empty_skeleton(bbox, pid))
                continue

            results = self._model(
                crop,
                device=self.device,
                conf=0.25,
                verbose=False,
            )

            best_skeleton = self._extract_best_pose(results, bbox, pid, x1, y1)
            skeletons.append(best_skeleton)

        return skeletons

    def _extract_best_pose(
        self,
        results: list,
        bbox: np.ndarray,
        pid: int,
        offset_x: int,
        offset_y: int,
    ) -> Skeleton:
        """Extract the highest-confidence pose from YOLO results."""
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            # Get the detection with highest confidence
            kpts_data = result.keypoints
            if hasattr(kpts_data, 'data') and len(kpts_data.data) > 0:
                # kpts_data.data shape: (N, 17, 3) — x, y, conf
                all_kpts = kpts_data.data.cpu().numpy()

                # Pick the person with highest average keypoint confidence
                best_idx = 0
                best_conf = 0.0
                for i in range(len(all_kpts)):
                    avg_conf = all_kpts[i, :, 2].mean()
                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_idx = i

                kpts = all_kpts[best_idx]  # (17, 3)

                # Map crop-local coordinates back to full frame
                keypoints = kpts[:, :2].copy()
                keypoints[:, 0] += offset_x
                keypoints[:, 1] += offset_y
                confidences = kpts[:, 2]

                return Skeleton(
                    keypoints=keypoints,
                    confidences=confidences,
                    bbox=np.array(bbox),
                    player_id=pid,
                )

        return self._empty_skeleton(bbox, pid)

    def _empty_skeleton(self, bbox: np.ndarray, pid: int) -> Skeleton:
        return Skeleton(
            keypoints=np.zeros((17, 2)),
            confidences=np.zeros(17),
            bbox=np.array(bbox),
            player_id=pid,
        )

    def _empty_skeletons(
        self, bboxes: list[np.ndarray], player_ids: list[int]
    ) -> list[Skeleton]:
        return [self._empty_skeleton(bbox, pid) for bbox, pid in zip(bboxes, player_ids)]
