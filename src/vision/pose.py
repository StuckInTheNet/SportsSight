"""Pose estimation — skeleton keypoint extraction from player crops."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch

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
    """Pose estimation using RTMPose (via MMPose) or fallback to MediaPipe."""

    def __init__(self, device: str = "cpu", model: str = "rtmpose-l") -> None:
        self.device = device
        self._model_name = model
        self._inferencer = None

    def load_model(self) -> None:
        """Load the pose estimation model."""
        try:
            from mmpose.apis import MMPoseInferencer
            self._inferencer = MMPoseInferencer(
                pose2d=self._model_name,
                device=self.device,
            )
            logger.info("Loaded MMPose model: %s on %s", self._model_name, self.device)
        except ImportError:
            logger.warning(
                "mmpose not available — pose estimation will be unavailable. "
                "Install with: pip install mmpose mmdet mmengine"
            )

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        player_ids: list[int],
    ) -> list[Skeleton]:
        """Estimate poses for detected players.

        Args:
            frame: Full BGR image
            bboxes: List of [x1, y1, x2, y2] bounding boxes
            player_ids: Corresponding player IDs

        Returns:
            List of Skeleton objects
        """
        if not bboxes:
            return []

        if self._inferencer is not None:
            return self._estimate_mmpose(frame, bboxes, player_ids)
        return self._estimate_fallback(frame, bboxes, player_ids)

    def _estimate_mmpose(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        player_ids: list[int],
    ) -> list[Skeleton]:
        """Run MMPose inference."""
        bbox_array = np.array(bboxes)

        results_gen = self._inferencer(
            frame,
            bboxes=bbox_array,
            show=False,
        )
        results = next(results_gen)

        skeletons: list[Skeleton] = []
        predictions = results.get("predictions", [[]])[0]

        for pred, bbox, pid in zip(predictions, bboxes, player_ids):
            kpts = np.array(pred["keypoints"])       # (17, 2)
            scores = np.array(pred["keypoint_scores"])  # (17,)

            skeletons.append(Skeleton(
                keypoints=kpts,
                confidences=scores,
                bbox=np.array(bbox),
                player_id=pid,
            ))

        return skeletons

    def _estimate_fallback(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        player_ids: list[int],
    ) -> list[Skeleton]:
        """Fallback: use OpenCV's DNN-based pose estimation or return empty."""
        logger.debug("No pose model loaded — returning empty skeletons")
        skeletons: list[Skeleton] = []
        for bbox, pid in zip(bboxes, player_ids):
            skeletons.append(Skeleton(
                keypoints=np.zeros((17, 2)),
                confidences=np.zeros(17),
                bbox=np.array(bbox),
                player_id=pid,
            ))
        return skeletons
