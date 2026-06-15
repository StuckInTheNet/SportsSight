"""Real-time inference engine — orchestrates the full pipeline for live streams."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from ..config import Config, load_config
from ..features.extractor import FeatureExtractor
from ..ingestion.sources import FramePacket, VideoSource
from ..models.fatigue import FatigueModel, FatigueScore
from ..vision.pipeline import VisionPipeline
from .alerts import AlertManager

logger = logging.getLogger(__name__)


class RealtimeEngine:
    """Orchestrates live video → fatigue scores with <2s latency target.

    Architecture:
    - Ingests frames from a live video source
    - Runs vision pipeline (detect → track → re-ID → pose)
    - Extracts features and scores fatigue
    - Publishes results to Redis Streams for WebSocket consumers
    - Manages alerts based on configurable thresholds

    The engine processes frames synchronously on the GPU but publishes
    results asynchronously to Redis, decoupling inference from delivery.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()

        # Pipeline components
        self.vision = VisionPipeline(self.config)
        self.features = FeatureExtractor(
            fps=self.config.pipeline.get("inference_fps", 15),
            windows=self.config.features.get("windows", [30, 120, 300]),
        )
        self.fatigue = FatigueModel(
            device=self.config.device,
            baseline_window_minutes=self.config.fatigue.get("baseline_window_minutes", 6),
        )
        self.alerts = AlertManager(
            thresholds=self.config.fatigue.get("thresholds", {}),
            cooldown=self.config.fatigue.get("alert_cooldown", 120),
        )

        self._redis: aioredis.Redis | None = None
        self._game_id: str | None = None
        self._running = False

        # Metrics
        self._frame_times: deque[float] = deque(maxlen=200)
        self._total_frames = 0

    async def initialize(self, require_redis: bool = True) -> None:
        """Load models and optionally connect to Redis."""
        logger.info("Initializing real-time engine...")
        self.vision.load_models()

        try:
            self._redis = aioredis.from_url(
                self.config.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Connected to Redis at %s", self.config.redis_url)
        except Exception as e:
            if require_redis:
                raise
            logger.warning("Redis unavailable (%s) — running in offline mode", e)
            self._redis = None

    async def start_game(self, game_id: str, source: VideoSource) -> None:
        """Begin processing a live game stream.

        Frame decoding + GPU inference run in a thread to avoid blocking the
        asyncio event loop. Results are published to Redis asynchronously.
        """
        self._game_id = game_id
        self._running = True
        self._total_frames = 0
        self._frame_times.clear()

        logger.info("Starting game %s from source %s", game_id, source.source_id)

        target_fps = self.config.pipeline.get("inference_fps", 15)

        # Producer: decode + infer in a thread, push results to a queue
        result_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()

        def _frame_worker():
            """Runs in a thread — decodes video and processes frames."""
            source.open()
            try:
                for packet in source.read_frames(target_fps=target_fps):
                    if not self._running:
                        break
                    start = time.monotonic()
                    scores = self._process_frame(packet)
                    elapsed = (time.monotonic() - start) * 1000
                    # Schedule queue put from the event loop thread
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put((scores, packet, elapsed)), loop
                    )
            finally:
                source.close()
                asyncio.run_coroutine_threadsafe(
                    result_queue.put(None), loop  # Sentinel
                )

        # Start the blocking worker in a thread
        worker_future = loop.run_in_executor(None, _frame_worker)

        try:
            while True:
                item = await result_queue.get()
                if item is None:
                    break

                scores, packet, elapsed = item
                self._frame_times.append(elapsed)
                self._total_frames += 1

                if scores:
                    await self._publish_scores(scores, packet)

                new_alerts = self.alerts.check(scores)
                if new_alerts:
                    await self._publish_alerts(new_alerts)

                if self._total_frames % 100 == 0:
                    avg = np.mean(self._frame_times[-100:])
                    max_lat = self.config.realtime.get("max_latency_ms", 2000)
                    status = "OK" if avg < max_lat else "SLOW"
                    logger.info(
                        "[%s] Frame %d | Avg latency: %.0fms | %s",
                        game_id, self._total_frames, avg, status,
                    )
        finally:
            self._running = False
            await asyncio.wrap_future(worker_future)
            logger.info("Game %s ended. Total frames: %d", game_id, self._total_frames)

    def _process_frame(self, packet: FramePacket) -> dict[int, FatigueScore]:
        """Run vision + features + fatigue on a single frame."""
        # Vision pipeline
        vision_result = self.vision.process_frame(packet)

        # Feature extraction
        features = self.features.process(vision_result)

        # Fatigue scoring
        scores = self.fatigue.update(features)

        return scores

    async def _publish_scores(
        self, scores: dict[int, FatigueScore], packet: FramePacket
    ) -> None:
        """Publish fatigue scores to Redis Stream."""
        if not self._redis or not self._game_id:
            return

        stream_key = f"sportssight:game:{self._game_id}:fatigue"
        max_len = self.config.realtime.get("stream_max_len", 10000)

        payload = {
            "game_id": self._game_id,
            "frame": packet.frame_number,
            "timestamp_ms": packet.timestamp_ms,
            "scores": json.dumps({
                str(pid): score.to_dict() for pid, score in scores.items()
            }),
        }

        await self._redis.xadd(stream_key, payload, maxlen=max_len)

    async def _publish_alerts(self, alerts: list[Any]) -> None:
        """Publish fatigue alerts to a separate Redis Stream."""
        if not self._redis or not self._game_id:
            return

        stream_key = f"sportssight:game:{self._game_id}:alerts"

        for alert in alerts:
            await self._redis.xadd(stream_key, alert.to_dict(), maxlen=1000)

    async def stop(self) -> None:
        """Stop the engine gracefully."""
        self._running = False
        if self._redis:
            await self._redis.aclose()

    async def process_recorded(self, game_id: str, source: VideoSource) -> dict[str, Any]:
        """Process a recorded game (non-live) and return full analysis.

        This is the legacy/historical analysis mode — processes the entire
        game and returns complete fatigue timelines.
        """
        self._game_id = game_id
        all_scores: list[dict[str, Any]] = []

        target_fps = self.config.pipeline.get("inference_fps", 15)
        source.open()

        try:
            for packet in source.read_frames(target_fps=target_fps):
                scores = self._process_frame(packet)
                if scores:
                    frame_data = {
                        "frame": packet.frame_number,
                        "timestamp_ms": packet.timestamp_ms,
                        "scores": {
                            pid: score.to_dict() for pid, score in scores.items()
                        },
                    }
                    all_scores.append(frame_data)

                    # Publish to Redis if connected
                    if self._redis:
                        await self._publish_scores(scores, packet)
        finally:
            source.close()

        return {
            "game_id": game_id,
            "total_frames": len(all_scores),
            "timeline": all_scores,
        }

    @property
    def avg_latency_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return float(np.mean(self._frame_times[-100:]))
