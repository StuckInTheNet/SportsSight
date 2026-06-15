"""Video source adapters — unified interface for files, broadcast, and camera feeds."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import av
import cv2
import numpy as np


@dataclass
class FramePacket:
    """Single frame with metadata from any video source."""

    frame: np.ndarray          # BGR image (H, W, 3)
    timestamp_ms: float        # Milliseconds from stream start
    frame_number: int
    source_id: str             # Identifies which camera/stream
    source_fps: float
    width: int
    height: int
    is_live: bool              # True for real-time streams


class VideoSource(ABC):
    """Base class for all video sources."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._is_open = False

    @abstractmethod
    def open(self) -> None:
        """Open the video source."""

    @abstractmethod
    def read_frames(self, target_fps: float | None = None) -> Generator[FramePacket, None, None]:
        """Yield frames from the source, optionally downsampled to target_fps."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    def is_open(self) -> bool:
        return self._is_open

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FileSource(VideoSource):
    """Local video file (MP4, AVI, MKV, etc.)."""

    def __init__(self, path: str | Path, source_id: str | None = None) -> None:
        self.path = Path(path)
        super().__init__(source_id or self.path.stem)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._is_open = True

    def read_frames(self, target_fps: float | None = None) -> Generator[FramePacket, None, None]:
        if not self._cap or not self._is_open:
            raise RuntimeError("Source not open. Call open() first.")

        frame_interval = 1
        if target_fps and target_fps < self._fps:
            frame_interval = max(1, int(round(self._fps / target_fps)))

        frame_num = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_num % frame_interval == 0:
                yield FramePacket(
                    frame=frame,
                    timestamp_ms=(frame_num / self._fps) * 1000.0,
                    frame_number=frame_num,
                    source_id=self.source_id,
                    source_fps=self._fps,
                    width=self._width,
                    height=self._height,
                    is_live=False,
                )
            frame_num += 1

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0.0

    def close(self) -> None:
        if self._cap:
            self._cap.release()
        self._is_open = False


class RTSPSource(VideoSource):
    """Proprietary camera feed via RTSP using PyAV for low-latency decoding."""

    def __init__(self, url: str, source_id: str = "camera") -> None:
        super().__init__(source_id)
        self.url = url
        self._container: av.container.InputContainer | None = None
        self._fps: float = 30.0

    def open(self) -> None:
        options = {
            "rtsp_transport": "tcp",
            "stimeout": "5000000",  # 5s connection timeout
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        self._container = av.open(self.url, options=options)
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        self._fps = float(stream.average_rate or 30)
        self._is_open = True

    def read_frames(self, target_fps: float | None = None) -> Generator[FramePacket, None, None]:
        if not self._container or not self._is_open:
            raise RuntimeError("Source not open. Call open() first.")

        frame_interval = 1
        if target_fps and target_fps < self._fps:
            frame_interval = max(1, int(round(self._fps / target_fps)))

        frame_num = 0
        for frame in self._container.decode(video=0):
            if frame_num % frame_interval == 0:
                img = frame.to_ndarray(format="bgr24")
                yield FramePacket(
                    frame=img,
                    timestamp_ms=float(frame.pts * frame.time_base * 1000) if frame.pts else 0.0,
                    frame_number=frame_num,
                    source_id=self.source_id,
                    source_fps=self._fps,
                    width=img.shape[1],
                    height=img.shape[0],
                    is_live=True,
                )
            frame_num += 1

    def close(self) -> None:
        if self._container:
            self._container.close()
        self._is_open = False


class RTMPSource(VideoSource):
    """Broadcast stream (RTMP / HLS) via PyAV."""

    def __init__(self, url: str, source_id: str = "broadcast") -> None:
        super().__init__(source_id)
        self.url = url
        self._container: av.container.InputContainer | None = None
        self._fps: float = 30.0

    def open(self) -> None:
        options = {"analyzeduration": "2000000", "probesize": "2000000"}
        self._container = av.open(self.url, options=options)
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        self._fps = float(stream.average_rate or 30)
        self._is_open = True

    def read_frames(self, target_fps: float | None = None) -> Generator[FramePacket, None, None]:
        if not self._container or not self._is_open:
            raise RuntimeError("Source not open. Call open() first.")

        frame_interval = 1
        if target_fps and target_fps < self._fps:
            frame_interval = max(1, int(round(self._fps / target_fps)))

        frame_num = 0
        wall_start = time.monotonic()
        for frame in self._container.decode(video=0):
            if frame_num % frame_interval == 0:
                img = frame.to_ndarray(format="bgr24")
                yield FramePacket(
                    frame=img,
                    timestamp_ms=(time.monotonic() - wall_start) * 1000.0,
                    frame_number=frame_num,
                    source_id=self.source_id,
                    source_fps=self._fps,
                    width=img.shape[1],
                    height=img.shape[0],
                    is_live=True,
                )
            frame_num += 1

    def close(self) -> None:
        if self._container:
            self._container.close()
        self._is_open = False
