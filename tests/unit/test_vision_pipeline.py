"""Tests for vision pipeline orchestrator."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.sources import FramePacket
from src.vision.pipeline import VisionPipeline, FrameResult


class TestFrameResult:
    def test_attributes(self):
        result = FrameResult(
            frame_number=42,
            timestamp_ms=2800.0,
            detections=[],
            tracks=[],
            player_ids={},
            skeletons=[],
            court_mapping=None,
            camera_cut=False,
        )
        assert result.frame_number == 42
        assert result.timestamp_ms == 2800.0
        assert result.camera_cut is False
        assert result.court_mapping is None


class TestVisionPipelineIoU:
    def test_compute_iou_identical(self):
        bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        iou = VisionPipeline._compute_iou(bbox, bbox)
        assert abs(iou - 1.0) < 1e-5

    def test_compute_iou_no_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        b = np.array([20, 20, 30, 30], dtype=np.float32)
        iou = VisionPipeline._compute_iou(a, b)
        assert iou < 1e-5

    def test_compute_iou_partial(self):
        a = np.array([0, 0, 20, 20], dtype=np.float32)
        b = np.array([10, 10, 30, 30], dtype=np.float32)
        iou = VisionPipeline._compute_iou(a, b)
        expected = 100.0 / 700.0
        assert abs(iou - expected) < 1e-5
