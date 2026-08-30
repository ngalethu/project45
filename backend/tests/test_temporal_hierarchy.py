from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.common.types import CandidateEvent, Detection, PoseResult  # noqa: E402
from app.edge.alert_manager import AlertManager  # noqa: E402
from app.edge.hierarchical_dms import merge_detections, run_hierarchical_dms, windshield_from_vehicle  # noqa: E402
from app.edge.behavior_rules import build_chest_roi  # noqa: E402
from app.edge.event_evidence import filter_dms_evidence  # noqa: E402
from app.cloud.media_detector import _select_video_probe_indices  # noqa: E402
from scripts.evaluate_sample_videos import _metrics, _predicted_events  # noqa: E402


class TemporalVotingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "rules": {
                "temporal_window_frames": 10,
                "temporal_cooldown_frames": 20,
                "phone_confirm_frames": 5,
                "no_seatbelt_confirm_frames": 7,
            }
        }

    def test_phone_votes_can_be_non_consecutive_inside_window(self):
        manager = AlertManager(self.config)
        candidate = CandidateEvent("using_phone", 0.8, (1, 2, 3, 4), 0.7, "test")
        sequence = [candidate, None, candidate, candidate, None, candidate, None, candidate]
        alerts = []
        for frame_index, item in enumerate(sequence, start=1):
            alerts.extend(manager.update(
                candidates=[item] if item else [],
                frame_index=frame_index,
                timestamp_iso="2026-01-01T00:00:00Z",
                current_time_sec=0.0,
                source_device="test",
            ))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].event_type, "using_phone")
        self.assertIn("temporal_votes=5/8", alerts[0].note)

    def test_window_must_be_between_five_and_fifteen(self):
        self.config["rules"]["temporal_window_frames"] = 4
        with self.assertRaises(ValueError):
            AlertManager(self.config)


class HierarchicalPipelineTests(unittest.TestCase):
    def test_windshield_is_upper_central_vehicle_region(self):
        roi = windshield_from_vehicle((100, 100, 500, 500), (600, 800, 3))
        self.assertEqual(roi, (148, 116, 452, 332))

    def test_merge_suppresses_same_class_overlap(self):
        detections = [
            Detection(0, "phone", 0.9, (10, 10, 50, 50)),
            Detection(0, "phone", 0.7, (12, 12, 49, 49)),
            Detection(1, "seatbelt", 0.8, (12, 12, 49, 49)),
        ]
        merged = merge_detections(detections)
        self.assertEqual(len(merged), 2)

    def test_vehicle_driver_and_chest_stages_execute(self):
        class VehicleDetector:
            def predict(self, frame, imgsz=640):
                return [Detection(2, "car", 0.9, (50, 40, 350, 260))]

        class DmsDetector:
            def predict(self, frame, imgsz=768):
                height, width = frame.shape[:2]
                return [Detection(1, "seatbelt", 0.8, (0, 0, max(8, width // 2), max(8, height // 2)))]

        class PoseEstimator:
            def predict(self, frame):
                height, width = frame.shape[:2]
                return PoseResult(points={
                    "left_shoulder": (width // 3, height // 3),
                    "right_shoulder": (2 * width // 3, height // 3),
                    "left_hip": (width // 3, 2 * height // 3),
                    "right_hip": (2 * width // 3, 2 * height // 3),
                    "nose": (width // 2, height // 5),
                })

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        result = run_hierarchical_dms(
            frame,
            dms_detector=DmsDetector(),
            pose_estimator=PoseEstimator(),
            vehicle_detector=VehicleDetector(),
            config={"chest_second_pass_enabled": True},
        )
        self.assertIn("vehicle", result.stages)
        self.assertIn("windshield", result.stages)
        self.assertIn("driver_roi", result.stages)
        self.assertIn("chest_second_pass", result.stages)
        self.assertTrue(result.detections)


class VideoMetricTests(unittest.TestCase):
    def test_no_confirmed_violation_is_normal(self):
        self.assertEqual(_predicted_events({"confirmed_events": []}), {"normal"})

    def test_zero_denominator_is_zero_not_fake_perfect_score(self):
        result = _metrics({"tp": 0, "fp": 0, "fn": 2, "tn": 3})
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)


class EventEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.pose = PoseResult(points={
            "nose": (100, 45),
            "left_ear": (88, 50),
            "right_ear": (112, 50),
            "left_shoulder": (70, 80),
            "right_shoulder": (130, 80),
            "left_wrist": (78, 60),
            "right_wrist": (140, 110),
            "left_hip": (75, 170),
            "right_hip": (125, 170),
        })
        self.frame_shape = (240, 320, 3)
        self.driver_roi = (40, 25, 180, 210)
        self.chest_roi = build_chest_roi(self.pose, 320, 240)

    def evidence(self, detections):
        return filter_dms_evidence(
            detections, self.pose, self.driver_roi, self.chest_roi, self.frame_shape
        )

    def test_rejects_giant_phone_box(self):
        result = self.evidence([Detection(0, "phone", 0.95, (0, 0, 310, 220))])
        self.assertNotIn("using_phone", result.event_confidence)
        self.assertEqual(result.rejected["phone_implausible_geometry"], 1)

    def test_rejects_phone_not_associated_with_driver(self):
        result = self.evidence([Detection(0, "phone", 0.90, (155, 180, 175, 210))])
        self.assertNotIn("using_phone", result.event_confidence)

    def test_accepts_phone_near_wrist_and_head(self):
        result = self.evidence([Detection(0, "phone", 0.82, (72, 50, 88, 70))])
        self.assertEqual(result.event_confidence["using_phone"], 0.82)

    def test_seatbelt_conflict_suppresses_no_seatbelt(self):
        detections = [
            Detection(1, "seatbelt", 0.84, (70, 75, 130, 165)),
            Detection(2, "no-seatbelt", 0.88, (65, 75, 135, 170)),
        ]
        result = self.evidence(detections)
        self.assertNotIn("no_seatbelt", result.event_confidence)

    def test_expanded_chest_roi_reaches_hips(self):
        x1, y1, x2, y2 = self.chest_roi
        self.assertLessEqual(x1, 50)
        self.assertGreaterEqual(x2, 150)
        self.assertGreaterEqual(y2, 170)


class AdaptiveVideoSamplingTests(unittest.TestCase):
    def test_scene_change_receives_a_probe_while_preserving_full_clip_anchors(self):
        class FakeCapture:
            def __init__(self):
                self.index = 0

            def set(self, _property, value):
                self.index = int(value)

            def read(self):
                value = 0 if self.index < 10 else 255
                return True, np.full((36, 64, 3), value, dtype=np.uint8)

        indices = _select_video_probe_indices(
            FakeCapture(), total_frames=20, budget=12, scene_candidates=20
        )
        self.assertEqual(len(indices), 12)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 19)
        self.assertIn(10, indices)


if __name__ == "__main__":
    unittest.main()
