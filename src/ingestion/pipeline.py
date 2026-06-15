"""Ingestion pipeline — manages multiple video sources and dispatches frames."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .sources import FramePacket, VideoSource

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Manages one or more video sources and routes frames to processors."""

    def __init__(self, target_fps: float = 15.0) -> None:
        self.target_fps = target_fps
        self._sources: dict[str, VideoSource] = {}
        self._processors: list[Callable[[FramePacket], Any]] = []

    def add_source(self, source: VideoSource) -> None:
        """Register a video source."""
        self._sources[source.source_id] = source
        logger.info("Added source: %s", source.source_id)

    def add_processor(self, processor: Callable[[FramePacket], Any]) -> None:
        """Register a frame processor (called for each frame in order)."""
        self._processors.append(processor)

    def run(self, source_id: str | None = None) -> None:
        """Process frames from one source (or all if source_id is None)."""
        sources = (
            [self._sources[source_id]] if source_id else list(self._sources.values())
        )

        for source in sources:
            logger.info("Starting ingestion from: %s", source.source_id)
            frame_count = 0

            with source:
                for packet in source.read_frames(target_fps=self.target_fps):
                    for processor in self._processors:
                        processor(packet)
                    frame_count += 1

                    if frame_count % 500 == 0:
                        logger.info(
                            "[%s] Processed %d frames (%.1fs)",
                            source.source_id,
                            frame_count,
                            packet.timestamp_ms / 1000.0,
                        )

            logger.info(
                "Finished %s: %d frames processed", source.source_id, frame_count
            )

    async def run_live(self, source_id: str) -> None:
        """Run a live source with async frame dispatch (for real-time inference)."""
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Unknown source: {source_id}")

        logger.info("Starting live ingestion from: %s", source.source_id)
        source.open()

        try:
            for packet in source.read_frames(target_fps=self.target_fps):
                for processor in self._processors:
                    processor(packet)
        finally:
            source.close()
