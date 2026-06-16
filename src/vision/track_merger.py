"""Post-game track merging — consolidates fragmented player identities.

After the pipeline runs, many player IDs are actually the same person split
across camera cuts. This module merges them by finding pairs of identities that:
1. Never co-exist in the same frame (temporal exclusivity)
2. Have similar jersey color histograms (appearance similarity)
3. Are on the same team (team constraint)

The result is a mapping from old player IDs to consolidated IDs.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrackMerger:
    """Merges fragmented player identities after game processing.

    Uses a greedy approach: for each identity, find the best merge candidate
    based on appearance similarity, constrained by temporal non-overlap
    and team assignment.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        min_track_frames: int = 15,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_track_frames = min_track_frames

    def merge(
        self,
        timeline: list[dict],
        identities: dict[int, dict],
    ) -> dict[int, int]:
        """Compute merge mapping from a game timeline.

        Args:
            timeline: List of frame entries, each with 'scores' dict (pid → score_data)
            identities: pid → {color_histogram, team, jersey_number, ...}

        Returns:
            Mapping of old_pid → new_pid (consolidated). PIDs not in the
            mapping keep their original ID.
        """
        # Step 1: Build frame-presence sets per player
        presence = self._build_presence(timeline)

        # Step 2: Build color histograms per player from their scores
        histograms = self._build_histograms(identities)

        # Step 3: Build team assignments
        teams = {pid: info.get("team", -1) for pid, info in identities.items()}

        # Step 4: Filter to significant tracks
        significant = {
            pid for pid, frames in presence.items()
            if len(frames) >= self.min_track_frames
        }
        logger.info("Track merger: %d significant tracks (>=%d frames)", len(significant), self.min_track_frames)

        # Step 5: Greedy merge
        merge_map: dict[int, int] = {}
        merged_into: dict[int, set[int]] = {}  # canonical_id → set of merged frame indices

        sorted_pids = sorted(significant, key=lambda p: len(presence[p]), reverse=True)

        for pid in sorted_pids:
            if pid in merge_map:
                continue  # Already merged into someone else

            # This PID is a canonical identity
            canonical_frames = set(presence[pid])
            if pid in merged_into:
                canonical_frames = merged_into[pid]

            pid_team = teams.get(pid, -1)
            pid_hist = histograms.get(pid)

            # Find merge candidates
            for other_pid in sorted_pids:
                if other_pid == pid or other_pid in merge_map:
                    continue

                other_frames = set(presence[other_pid])

                # Constraint 1: Temporal exclusivity (no frame overlap)
                overlap = canonical_frames & other_frames
                if len(overlap) > 2:  # Allow tiny overlap from tracker noise
                    continue

                # Constraint 2: Same team (or unknown)
                other_team = teams.get(other_pid, -1)
                if pid_team >= 0 and other_team >= 0 and pid_team != other_team:
                    continue

                # Constraint 3: Appearance similarity
                other_hist = histograms.get(other_pid)
                if pid_hist is not None and other_hist is not None:
                    sim = self._histogram_similarity(pid_hist, other_hist)
                    if sim < self.similarity_threshold:
                        continue
                else:
                    # No histograms — only merge if same team is confirmed
                    if pid_team < 0 or other_team < 0:
                        continue

                # Merge other_pid into pid
                merge_map[other_pid] = pid
                canonical_frames |= other_frames
                if pid not in merged_into:
                    merged_into[pid] = canonical_frames
                else:
                    merged_into[pid] = canonical_frames

        # Count merges
        if merge_map:
            canonical_count = len(significant) - len(merge_map)
            logger.info(
                "Track merger: merged %d tracks → %d consolidated identities (was %d)",
                len(merge_map), canonical_count, len(significant),
            )
        else:
            logger.info("Track merger: no merge candidates found")

        return merge_map

    def apply_merge(
        self, timeline: list[dict], merge_map: dict[int, int]
    ) -> list[dict]:
        """Apply merge mapping to a timeline, rewriting player IDs."""
        if not merge_map:
            return timeline

        merged_timeline = []
        for entry in timeline:
            new_scores = {}
            for pid_str, score in entry["scores"].items():
                pid = int(pid_str) if isinstance(pid_str, str) else pid_str
                canonical = merge_map.get(pid, pid)
                canonical_str = str(canonical)

                if canonical_str in new_scores:
                    # Duplicate in same frame after merge — keep higher confidence
                    existing = new_scores[canonical_str]
                    if score.get("confidence", 0) > existing.get("confidence", 0):
                        new_scores[canonical_str] = score
                else:
                    score_copy = dict(score)
                    score_copy["player_id"] = canonical
                    new_scores[canonical_str] = score_copy

            merged_entry = dict(entry)
            merged_entry["scores"] = new_scores
            merged_timeline.append(merged_entry)

        return merged_timeline

    def _build_presence(self, timeline: list[dict]) -> dict[int, set[int]]:
        """Build frame-index presence sets per player ID."""
        presence: dict[int, set[int]] = defaultdict(set)
        for i, entry in enumerate(timeline):
            for pid_str in entry.get("scores", {}):
                pid = int(pid_str) if isinstance(pid_str, str) else pid_str
                presence[pid].add(i)
        return dict(presence)

    def _build_histograms(self, identities: dict[int, dict]) -> dict[int, np.ndarray]:
        """Extract color histograms from identity data."""
        histograms = {}
        for pid, info in identities.items():
            hist = info.get("color_histogram")
            if hist is not None:
                if isinstance(hist, list):
                    hist = np.array(hist, dtype=np.float32)
                histograms[pid] = hist
        return histograms

    @staticmethod
    def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute histogram similarity using correlation."""
        return float(cv2.compareHist(
            a.astype(np.float32),
            b.astype(np.float32),
            cv2.HISTCMP_CORREL,
        ))
