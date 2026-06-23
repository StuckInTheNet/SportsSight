"""Jersey number detection — OCR-based player identification from broadcast video.

Extracts jersey numbers from player crops using EasyOCR. This provides a stable
identity signal across camera cuts, unlike appearance-based ReID which breaks
when the broadcast angle changes.

Strategy:
1. Crop the upper-back/chest region of each player (where jersey numbers appear)
2. Preprocess: enhance contrast, threshold for white/dark text on jersey
3. Run OCR, filter for 1-2 digit numbers in valid NBA range (0-99)
4. Maintain a confidence-weighted vote per track to handle noisy reads
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Valid NBA jersey numbers
VALID_JERSEY_NUMBERS = set(range(0, 100))


@dataclass
class JerseyRead:
    """A single jersey number OCR read."""

    number: int
    confidence: float
    frame_number: int


class JerseyDetector:
    """Detects jersey numbers from player crops using EasyOCR."""

    def __init__(self, device: str = "cpu") -> None:
        self._reader = None
        self._device = device
        # Per-track accumulator: track_id → list of reads
        self._track_reads: dict[int, list[JerseyRead]] = defaultdict(list)
        # Resolved: track_id → confirmed jersey number
        self._track_jersey: dict[int, int] = {}
        # How many frames between OCR attempts per track (saves compute)
        self.ocr_interval = 15
        self._track_last_ocr: dict[int, int] = {}

    def load_model(self) -> None:
        """Load EasyOCR reader."""
        try:
            import easyocr
            gpu = self._device in ("cuda", "mps")
            self._reader = easyocr.Reader(
                ["en"],
                gpu=gpu,
                verbose=False,
            )
            logger.info("EasyOCR loaded (gpu=%s)", gpu)
        except ImportError:
            logger.warning("easyocr not installed — jersey detection disabled")
        except Exception as e:
            logger.warning("EasyOCR failed to load: %s", e)

    def detect(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_number: int,
    ) -> int | None:
        """Attempt to read jersey number from a player crop.

        Returns the jersey number if confidently identified, else None.
        Uses a vote-based system: multiple reads across frames are accumulated,
        and the most common valid number wins once it has enough votes.
        """
        # Return cached result if we've already resolved this track
        if track_id in self._track_jersey:
            return self._track_jersey[track_id]

        # Throttle OCR — don't run every frame
        last_ocr = self._track_last_ocr.get(track_id, -999)
        if frame_number - last_ocr < self.ocr_interval:
            return self._get_best_candidate(track_id)

        self._track_last_ocr[track_id] = frame_number

        if self._reader is None or crop.size == 0:
            return None

        # Extract jersey region (upper torso: 15-60% height, center 70% width)
        h, w = crop.shape[:2]
        if h < 40 or w < 20:
            return None

        y_start = int(h * 0.15)
        y_end = int(h * 0.55)
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)
        jersey_region = crop[y_start:y_end, x_start:x_end]

        if jersey_region.size == 0:
            return None

        # Preprocess for OCR
        processed = self._preprocess_jersey(jersey_region)

        # Run OCR
        number = self._run_ocr(processed)
        if number is not None:
            self._track_reads[track_id].append(
                JerseyRead(number=number, confidence=0.8, frame_number=frame_number)
            )

            # Also try on the original (unprocessed) region
            number_raw = self._run_ocr(jersey_region)
            if number_raw is not None and number_raw != number:
                self._track_reads[track_id].append(
                    JerseyRead(number=number_raw, confidence=0.6, frame_number=frame_number)
                )

        # Check if we have enough votes to confirm
        return self._resolve_jersey(track_id)

    def _preprocess_jersey(self, region: np.ndarray) -> np.ndarray:
        """Enhance jersey number visibility for OCR."""
        # Upscale small crops for better OCR
        h, w = region.shape[:2]
        if h < 60:
            scale = 60 / h
            region = cv2.resize(
                region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Adaptive threshold to handle varying jersey colors
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        return thresh

    def _run_ocr(self, image: np.ndarray) -> int | None:
        """Run EasyOCR and extract a valid jersey number."""
        try:
            results = self._reader.readtext(
                image,
                allowlist="0123456789",
                paragraph=False,
                min_size=10,
            )

            for bbox, text, conf in results:
                text = text.strip()
                if not text:
                    continue

                # Must be 1-2 digits
                if not text.isdigit() or len(text) > 2:
                    continue

                number = int(text)
                if number in VALID_JERSEY_NUMBERS and conf > 0.3:
                    return number

        except Exception as e:
            logger.debug("OCR failed: %s", e)

        return None

    def _resolve_jersey(self, track_id: int) -> int | None:
        """Check if we have enough consistent reads to confirm a jersey number."""
        reads = self._track_reads.get(track_id, [])
        if len(reads) < 2:
            return self._get_best_candidate(track_id)

        # Count votes weighted by confidence
        votes: Counter = Counter()
        for read in reads:
            votes[read.number] += read.confidence

        if not votes:
            return None

        best_number, best_score = votes.most_common(1)[0]

        # Need at least 2 agreeing reads with combined confidence > 1.0
        if best_score >= 1.0 and votes[best_number] >= 1.0:
            self._track_jersey[track_id] = best_number
            logger.debug(
                "Track %d → jersey #%d (score=%.1f, reads=%d)",
                track_id, best_number, best_score, len(reads),
            )
            return best_number

        return self._get_best_candidate(track_id)

    def _get_best_candidate(self, track_id: int) -> int | None:
        """Return the current best-guess jersey number (not yet confirmed)."""
        if track_id in self._track_jersey:
            return self._track_jersey[track_id]
        reads = self._track_reads.get(track_id, [])
        if not reads:
            return None
        votes: Counter = Counter()
        for read in reads:
            votes[read.number] += read.confidence
        if votes:
            return votes.most_common(1)[0][0]
        return None

    def get_resolved_jerseys(self) -> dict[int, int]:
        """Return all confirmed track_id → jersey_number mappings."""
        return dict(self._track_jersey)

    def reset(self) -> None:
        """Clear all state between games."""
        self._track_reads.clear()
        self._track_jersey.clear()
        self._track_last_ocr.clear()
