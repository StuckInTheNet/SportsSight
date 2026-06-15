"""Tests for player re-identification module."""

import numpy as np
import pytest

from src.vision.reid import PlayerReID, PlayerIdentity


class TestPlayerIdentity:
    def test_mean_embedding_empty(self):
        identity = PlayerIdentity(player_id=1)
        assert identity.mean_embedding is None

    def test_mean_embedding_single(self):
        identity = PlayerIdentity(player_id=1)
        emb = np.array([1.0, 2.0, 3.0])
        identity.add_embedding(emb)
        mean = identity.mean_embedding
        assert mean is not None
        np.testing.assert_array_almost_equal(mean, emb)

    def test_mean_embedding_multiple(self):
        identity = PlayerIdentity(player_id=1)
        identity.add_embedding(np.array([1.0, 0.0]))
        identity.add_embedding(np.array([0.0, 1.0]))
        mean = identity.mean_embedding
        np.testing.assert_array_almost_equal(mean, [0.5, 0.5])

    def test_gallery_bounded(self):
        identity = PlayerIdentity(player_id=1)
        for i in range(20):
            identity.add_embedding(np.array([float(i)]), max_gallery=5)
        assert len(identity.embeddings) == 5
        # Should keep the last 5
        assert identity.embeddings[0][0] == 15.0

    def test_track_ids_accumulated(self):
        identity = PlayerIdentity(player_id=1)
        identity.track_ids.append(10)
        identity.track_ids.append(20)
        assert identity.track_ids == [10, 20]


class TestPlayerReID:
    def test_color_histogram_shape(self):
        reid = PlayerReID(device="cpu")
        # Create a fake player crop (100x50x3 BGR)
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        hist = reid.extract_color_histogram(crop)
        # 30 hue bins * 32 saturation bins = 960
        assert hist.shape == (960,)

    def test_color_histogram_normalized(self):
        reid = PlayerReID(device="cpu")
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        hist = reid.extract_color_histogram(crop)
        # Normalized histograms should have values in [0, 1]
        assert hist.max() <= 1.0 + 1e-6

    def test_camera_cut_detection_no_prev(self):
        reid = PlayerReID(device="cpu")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert reid.detect_camera_cut(frame, None) is False

    def test_camera_cut_detection_same_frame(self):
        reid = PlayerReID(device="cpu")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert reid.detect_camera_cut(frame, frame) is False

    def test_camera_cut_detection_different_frames(self):
        reid = PlayerReID(device="cpu")
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.full((480, 640, 3), 255, dtype=np.uint8)
        # Completely different frames should trigger a camera cut
        assert reid.detect_camera_cut(frame2, frame1) is True

    def test_match_track_creates_new_identity(self):
        reid = PlayerReID(device="cpu")
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        pid = reid.match_track(track_id=1, crop=crop, frame_number=0)
        assert pid == 1
        assert reid.get_identity(pid) is not None
        assert 1 in reid.get_identity(pid).track_ids

    def test_match_track_reuses_identity_without_cut(self):
        reid = PlayerReID(device="cpu")
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        pid1 = reid.match_track(track_id=1, crop=crop, frame_number=0)
        pid2 = reid.match_track(track_id=1, crop=crop, frame_number=1)
        assert pid1 == pid2

    def test_reset_clears_state(self):
        reid = PlayerReID(device="cpu")
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        reid.match_track(track_id=1, crop=crop, frame_number=0)
        assert len(reid.get_all_identities()) == 1

        reid.reset()
        assert len(reid.get_all_identities()) == 0

    def test_extract_embedding_without_model(self):
        reid = PlayerReID(device="cpu")
        # Model not loaded — should return None
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        assert reid.extract_embedding(crop) is None

    def test_extract_embedding_tiny_crop(self):
        reid = PlayerReID(device="cpu")
        # Tiny crop should be rejected
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        assert reid.extract_embedding(tiny) is None

    def test_compute_similarity_color_only(self):
        """Without embeddings, similarity should be based on color histograms."""
        reid = PlayerReID(device="cpu")
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        hist = reid.extract_color_histogram(crop)

        identity = PlayerIdentity(player_id=1, color_histogram=hist)
        score = reid._compute_similarity(None, hist, identity)
        # Same histogram should give high similarity
        assert score > 0.8
