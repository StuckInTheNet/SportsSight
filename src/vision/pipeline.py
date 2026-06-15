"""Vision pipeline orchestrator — chains detection → tracking → re-ID → pose."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..ingestion.sources import FramePacket
from .court import CourtDetector, CourtMapping
from .detector import Detection, PlayerDetector
from .pose import PoseEstimator, Skeleton
from .reid import PlayerReID
from .tracker import ByteTracker, Track

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Complete vision output for a single frame."""

    frame_number: int
    timestamp_ms: float
    detections: list[Detection]
    tracks: list[Track]
    player_ids: dict[int, int]    # track_id → player_id
    skeletons: list[Skeleton]
    court_mapping: CourtMapping | None
    camera_cut: bool


class VisionPipeline:
    """Full vision pipeline: detect → track → re-identify → estimate pose → map court."""

    def __init__(self, config: Config) -> None:
        self.config = config

        # Initialize components
        self.detector = PlayerDetector(
            model_path=config.detection.get("model", "yolov8x.pt"),
            device=config.device,
            confidence=config.detection.get("confidence_threshold", 0.5),
            nms_iou=config.detection.get("nms_iou", 0.45),
            min_area=config.detection.get("min_area", 2000),
        )

        self.tracker = ByteTracker(
            high_thresh=config.tracking.get("track_high_thresh", 0.6),
            low_thresh=config.tracking.get("track_low_thresh", 0.1),
            match_thresh=config.tracking.get("match_thresh", 0.8),
            track_buffer=config.tracking.get("track_buffer", 60),
        )

        self.reid = PlayerReID(
            model_name=config.reid.get("model", "osnet_x1_0"),
            device=config.device,
            match_threshold=config.reid.get("match_threshold", 0.65),
            gallery_size=config.reid.get("gallery_size", 10),
        )

        self.pose = PoseEstimator(
            device=config.device,
            model=config.pose.get("model", "yolov8x-pose"),
        )

        self.court = CourtDetector(
            update_interval=config.court.get("update_interval", 300),
            min_points=config.court.get("min_lines", 4),
        )

        self._prev_frame: np.ndarray | None = None
        self._frame_count = 0

    def load_models(self) -> None:
        """Load all ML models. Call once before processing."""
        logger.info("Loading vision models on device: %s", self.config.device)
        self.reid.load_model()
        self.pose.load_model()
        logger.info("All vision models loaded")

    def process_frame(self, packet: FramePacket) -> FrameResult:
        """Run the full vision pipeline on one frame."""
        frame = packet.frame
        frame_num = packet.frame_number

        # 1. Detect camera cuts (triggers re-ID matching)
        camera_cut = self.reid.detect_camera_cut(frame, self._prev_frame)
        if camera_cut:
            logger.info("Camera cut detected at frame %d", frame_num)

        # 2. Detect players
        detections = self.detector.detect(frame)

        # 3. Track players (maintains IDs within continuous footage)
        tracks = self.tracker.update(detections)

        # 4. Re-identify across camera cuts
        player_ids: dict[int, int] = {}
        for track in tracks:
            # Find the detection crop closest to this track
            crop = self._get_track_crop(track, detections, frame)
            if crop is not None and crop.size > 0:
                pid = self.reid.match_track(track.track_id, crop, frame_num)
                player_ids[track.track_id] = pid

        # 5. Estimate poses
        bboxes = [t.bbox for t in tracks if t.track_id in player_ids]
        pids = [player_ids[t.track_id] for t in tracks if t.track_id in player_ids]
        skeletons = self.pose.estimate(frame, bboxes, pids)

        # 6. Update court homography (periodically)
        court_mapping = self.court.detect_and_map(frame, frame_num)

        self._prev_frame = frame
        self._frame_count += 1

        return FrameResult(
            frame_number=frame_num,
            timestamp_ms=packet.timestamp_ms,
            detections=detections,
            tracks=tracks,
            player_ids=player_ids,
            skeletons=skeletons,
            court_mapping=court_mapping,
            camera_cut=camera_cut,
        )

    def _get_track_crop(
        self, track: Track, detections: list[Detection], frame: np.ndarray
    ) -> np.ndarray | None:
        """Get the best crop for a track — either from matched detection or from bbox."""
        # Try to find the detection that overlaps most with this track
        best_iou = 0.0
        best_crop = None
        for det in detections:
            iou = self._compute_iou(track.bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_crop = det.crop

        if best_crop is not None and best_iou > 0.5:
            return best_crop

        # Fallback: crop from track bbox directly
        x1, y1, x2, y2 = track.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2].copy()
        return None

    @staticmethod
    def _compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)
