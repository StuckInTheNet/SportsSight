"""Player detection using YOLOv8."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected player bounding box."""

    bbox: np.ndarray       # [x1, y1, x2, y2] in pixels
    confidence: float
    class_id: int          # COCO class (0 = person)
    crop: np.ndarray       # Cropped image of the player (for re-ID)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PlayerDetector:
    """YOLOv8-based player detection."""

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        device: str = "cpu",
        confidence: float = 0.5,
        nms_iou: float = 0.45,
        min_area: float = 2000.0,
    ) -> None:
        self.device = device
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.min_area = min_area

        logger.info("Loading YOLO model: %s on %s", model_path, device)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect players in a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            List of Detection objects for persons found in the frame.
        """
        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence,
            iou=self.nms_iou,
            classes=[0],  # Person only
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(boxes.conf[i].cpu())
                cls = int(boxes.cls[i].cpu())

                # Filter small detections (likely spectators or distant refs)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w * h < self.min_area:
                    continue

                # Aspect ratio filter — players are roughly 1:2 to 1:4 ratio
                aspect = h / max(w, 1)
                if aspect < 1.2 or aspect > 5.0:
                    continue

                # Extract crop for re-ID
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2].copy()

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls,
                    crop=crop,
                ))

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect players in a batch of frames."""
        results = self.model(
            frames,
            device=self.device,
            conf=self.confidence,
            iou=self.nms_iou,
            classes=[0],
            verbose=False,
        )

        all_detections: list[list[Detection]] = []
        for frame, result in zip(frames, results):
            frame_detections: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                    conf = float(boxes.conf[i].cpu())
                    cls = int(boxes.cls[i].cpu())

                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w * h < self.min_area:
                        continue
                    aspect = h / max(w, 1)
                    if aspect < 1.2 or aspect > 5.0:
                        continue

                    x1, y1, x2, y2 = bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2].copy()

                    frame_detections.append(Detection(
                        bbox=bbox, confidence=conf, class_id=cls, crop=crop,
                    ))
            all_detections.append(frame_detections)

        return all_detections
