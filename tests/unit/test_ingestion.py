"""Tests for video ingestion sources."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.sources import FileSource, FramePacket
from src.ingestion.pipeline import IngestionPipeline


class TestFramePacket:
    def test_attributes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            frame=frame,
            timestamp_ms=1000.0,
            frame_number=30,
            source_id="test",
            source_fps=30.0,
            width=640,
            height=480,
            is_live=False,
        )
        assert packet.width == 640
        assert packet.height == 480
        assert packet.timestamp_ms == 1000.0
        assert not packet.is_live


class TestFileSource:
    def test_nonexistent_file_raises(self):
        source = FileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError):
            source.open()

    def test_context_manager(self):
        source = FileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError):
            with source:
                pass

    def test_read_without_open_raises(self):
        source = FileSource("/some/path.mp4")
        with pytest.raises(RuntimeError, match="not open"):
            next(source.read_frames())

    def test_properties_before_open(self):
        source = FileSource("/some/path.mp4")
        assert source.source_id == "path"
        assert not source.is_open


class TestIngestionPipeline:
    def test_add_source(self):
        pipeline = IngestionPipeline(target_fps=15)
        source = MagicMock()
        source.source_id = "test"
        pipeline.add_source(source)
        assert "test" in pipeline._sources

    def test_add_processor(self):
        pipeline = IngestionPipeline()
        processor = MagicMock()
        pipeline.add_processor(processor)
        assert len(pipeline._processors) == 1
