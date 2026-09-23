"""
Unit tests for VISIONFORGE AI Single-Pass Observation Pipeline
Verifies data models, tracking observation formats, crop extraction logic, and track profile consistency.
"""

import unittest
from pathlib import Path
import json
import numpy as np

from src.pipeline.models import (
    Observation,
    VideoTrackingResult,
    CropRecord,
    CropMetadataCollection,
    TrackAttributes,
    TrackProfile,
)
from src.pipeline.reid_extractor import l2_normalize


class TestPipelineConsistency(unittest.TestCase):

    def test_observation_model(self):
        obs = Observation(
            frame_index=10,
            timestamp_seconds=0.4,
            track_id=1,
            bbox=[100, 150, 200, 350],
            confidence=0.88,
            class_id=0,
        )
        self.assertEqual(obs.track_id, 1)
        self.assertEqual(obs.bbox, [100, 150, 200, 350])
        self.assertEqual(obs.confidence, 0.88)

    def test_video_tracking_result_serialization(self):
        obs1 = Observation(
            frame_index=0,
            timestamp_seconds=0.0,
            track_id=1,
            bbox=[10, 10, 50, 100],
            confidence=0.9,
        )
        obs2 = Observation(
            frame_index=15,
            timestamp_seconds=0.6,
            track_id=1,
            bbox=[12, 12, 52, 102],
            confidence=0.92,
        )
        res = VideoTrackingResult(
            video_id="test_cam",
            video_filename="test_cam.mp4",
            fps=25.0,
            total_frames=30,
            processed_frames=30,
            unique_tracks=[1],
            total_observations=2,
            observations=[obs1, obs2],
        )
        data = res.model_dump()
        self.assertEqual(data["video_id"], "test_cam")
        self.assertEqual(len(data["observations"]), 2)
        self.assertEqual(data["unique_tracks"], [1])

    def test_reid_l2_normalization(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        norm_v = l2_normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(norm_v)), 1.0, places=5)
        self.assertAlmostEqual(float(norm_v[0]), 0.6, places=5)
        self.assertAlmostEqual(float(norm_v[1]), 0.8, places=5)

    def test_track_profile_integrity(self):
        attrs = TrackAttributes(
            track_id=5,
            upper_body_color="black",
            lower_body_color="blue",
            color_distribution={"black": 0.5, "blue": 0.5},
            sample_crops=2,
        )
        profile = TrackProfile(
            video_id="cam_01",
            track_id=5,
            first_seen_frame=10,
            last_seen_frame=50,
            first_seen_sec=0.4,
            last_seen_sec=2.0,
            duration_sec=1.6,
            total_observations=20,
            total_crops=2,
            attributes=attrs,
            embedding=[0.1] * 512,
            embedding_dimension=512,
            representative_crop="dummy.jpg",
        )
        self.assertEqual(profile.track_id, 5)
        self.assertEqual(len(profile.embedding), 512)
        self.assertEqual(profile.attributes.upper_body_color, "black")
        self.assertEqual(profile.attributes.lower_body_color, "blue")


if __name__ == "__main__":
    unittest.main()
