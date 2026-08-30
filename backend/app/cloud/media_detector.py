from __future__ import annotations
import os
import time
from collections import deque
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from app.edge.yolo_detector import YoloDetector
from app.edge.pose_estimator import PoseEstimator
from app.edge.hierarchical_dms import run_hierarchical_dms
from app.edge.event_evidence import filter_dms_evidence

_DETECTOR = None
_VEHICLE_DETECTOR = None
_POSE_ESTIMATOR = None

HIERARCHY_CONFIG = {
    "vehicle_stage_enabled": True,
    "vehicle_names": ["car", "truck", "bus"],
    "max_vehicles": 3,
    "min_vehicle_area_ratio": 0.025,
    "windshield_x_margin": 0.12,
    "windshield_y_start": 0.04,
    "windshield_y_end": 0.58,
    "vehicle_imgsz": 640,
    "dms_imgsz": 768,
    "chest_imgsz": 768,
    "chest_second_pass_enabled": True,
    "video_probe_budget": 48,
    "video_scene_candidates": 96,
}


def _select_video_probe_indices(
    cap: cv2.VideoCapture,
    total_frames: int,
    budget: int = 48,
    scene_candidates: int = 96,
) -> List[int]:
    """Cover the full clip and spend half the budget near visual scene changes."""
    if total_frames <= 0:
        return list(range(max(1, budget)))
    budget = max(12, min(int(budget), total_frames))
    candidate_count = max(budget, min(int(scene_candidates), total_frames))
    candidate_indices = np.linspace(0, total_frames - 1, candidate_count, dtype=int)

    change_scores: list[tuple[float, int]] = []
    previous = None
    for index in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        tiny = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
        score = float(cv2.absdiff(gray, previous).mean()) if previous is not None else 255.0
        change_scores.append((score, int(index)))
        previous = gray

    anchor_count = max(6, budget // 2)
    selected = set(np.linspace(0, total_frames - 1, anchor_count, dtype=int).tolist())
    for _, index in sorted(change_scores, reverse=True):
        selected.add(index)
        if len(selected) >= budget:
            break
    return sorted(selected)[:budget]

def get_yolo_detector() -> YoloDetector:
    global _DETECTOR
    if _DETECTOR is None:
        possible_paths = [
            os.path.join("models", "best.pt"),
            os.path.join("backend", "models", "best.pt"),
            os.path.join(Path(__file__).parents[2], "models", "best.pt"),
            os.path.join(Path(__file__).parents[1], "models", "best.pt"),
        ]
        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break
                
        if not model_path:
            raise RuntimeError(f"YOLO model not found in paths: {possible_paths}")
            
        print(f"[MEDIA DETECTOR] Loading YOLO model from: {model_path}")
        _DETECTOR = YoloDetector(model_path=model_path, conf=0.35, iou=0.45)
    return _DETECTOR

def get_pose_estimator() -> PoseEstimator:
    global _POSE_ESTIMATOR
    if _POSE_ESTIMATOR is None:
        print("[MEDIA DETECTOR] Loading MediaPipe Pose Estimator...")
        _POSE_ESTIMATOR = PoseEstimator(min_detection_confidence=0.4, min_visibility=0.3)
    return _POSE_ESTIMATOR


def get_vehicle_detector() -> YoloDetector | None:
    global _VEHICLE_DETECTOR
    if _VEHICLE_DETECTOR is not None:
        return _VEHICLE_DETECTOR
    candidates = [
        Path(__file__).resolve().parents[2] / "yolo11m.pt",
        Path("yolo11m.pt"),
        Path("backend") / "yolo11m.pt",
    ]
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        return None
    _VEHICLE_DETECTOR = YoloDetector(str(model_path), conf=0.30, iou=0.50)
    return _VEHICLE_DETECTOR


def _event_for_detection(class_name: str) -> str | None:
    canonical = class_name.lower().strip().replace("_", "-").replace(" ", "-")
    if canonical == "phone":
        return "using_phone"
    if canonical == "no-seatbelt":
        return "no_seatbelt"
    if canonical == "seatbelt":
        return "seatbelt"
    return None

def draw_dual_engine_geometry(frame: np.ndarray, pose_points: Dict[str, tuple[int, int]]) -> bool:
    """
    Vẽ Vùng Ngực (Chest ROI) nối 2 Vai & 2 Hông + Các điểm khớp tay/đầu.
    Trả về True nếu xác định được Chest ROI.
    """
    ls = pose_points.get("left_shoulder")
    rs = pose_points.get("right_shoulder")
    lh = pose_points.get("left_hip")
    rh = pose_points.get("right_hip")

    has_chest_roi = False
    if ls and rs:
        # Drawing shoulder line
        cv2.line(frame, ls, rs, (255, 200, 0), 2)
        
        if lh and rh:
            # Drawing Chest ROI quadrilateral (Vai trái -> Vai phải -> Hông phải -> Hông trái)
            chest_poly = np.array([ls, rs, rh, lh], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [chest_poly], isClosed=True, color=(255, 191, 0), thickness=2)
            
            # Semi-transparent overlay for Chest ROI
            overlay = frame.copy()
            cv2.fillPoly(overlay, [chest_poly], color=(255, 215, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            has_chest_roi = True

    # Draw Ear and Wrist Keypoints
    for kpt in ["left_ear", "right_ear", "nose", "left_wrist", "right_wrist"]:
        pt = pose_points.get(kpt)
        if pt:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)

    return has_chest_roi

def process_uploaded_image(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image file format")

    hierarchy = run_hierarchical_dms(
        frame,
        dms_detector=get_yolo_detector(),
        pose_estimator=get_pose_estimator(),
        vehicle_detector=get_vehicle_detector(),
        config=HIERARCHY_CONFIG,
    )
    evidence = filter_dms_evidence(
        hierarchy.detections,
        hierarchy.pose,
        hierarchy.driver_roi,
        hierarchy.chest_roi,
        frame.shape,
    )
    detections = evidence.accepted
    draw_dual_engine_geometry(frame, hierarchy.pose.points)

    detected_events = []
    max_conf = 0.0
    primary_event = "normal"

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        c_name = det.class_name.lower()
        conf = det.confidence
        event = _event_for_detection(c_name)
        if event is None:
            continue
        color = (0, 0, 239) if event == "using_phone" else ((0, 215, 255) if event == "no_seatbelt" else (34, 197, 94))

        if event in ["using_phone", "no_seatbelt"]:
            detected_events.append(event)
            if conf > max_conf:
                primary_event = event
                max_conf = conf

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{c_name} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    timestamp_str = int(time.time() * 1000)
    out_dir = Path("outputs/cloud_uploads/frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"auto_detect_{timestamp_str}.jpg"
    out_path = out_dir / out_filename
    cv2.imwrite(str(out_path), frame)

    return {
        "event_type": primary_event,
        "confidence": round(max_conf, 3),
        "detections_count": len(detections),
        "detected_labels": [d.class_name for d in detections],
        "pipeline_stages": hierarchy.stages,
        "rejected_evidence": evidence.rejected,
        "frame_path": str(out_path).replace("\\", "/"),
        "frame_url": f"/uploads/frames/{out_filename}",
    }

def process_uploaded_video(file_path: str, max_frames: int = 150) -> Dict[str, Any]:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Invalid video file format")

    detector = get_yolo_detector()
    pose_estimator = get_pose_estimator()
    vehicle_detector = get_vehicle_detector()

    timestamp_str = int(time.time() * 1000)
    out_dir = Path("outputs/cloud_uploads/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"auto_detect_{timestamp_str}.mp4"
    out_path = out_dir / out_filename

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    probe_budget = min(max_frames, int(HIERARCHY_CONFIG.get("video_probe_budget", 48)))
    probe_indices = _select_video_probe_indices(
        cap,
        source_total_frames,
        budget=probe_budget,
        scene_candidates=int(HIERARCHY_CONFIG.get("video_scene_candidates", 96)),
    )

    out = None
    codecs_to_try = [("avc1", "mp4"), ("H264", "mp4"), ("mp4v", "mp4")]
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_path), fourcc, min(fps, 10.0), (width, height))
            if writer.isOpened():
                out = writer
                break
        except Exception:
            continue
            
    if out is None or not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, min(fps, 10.0), (width, height))

    violation_events = ("using_phone", "no_seatbelt")
    detected_event_counts: Dict[str, int] = {event: 0 for event in violation_events}
    event_peak_conf: Dict[str, float] = {event: 0.0 for event in violation_events}
    event_histories = {event: deque(maxlen=12) for event in violation_events}
    min_hits = {"using_phone": 5, "no_seatbelt": 6}
    min_mean_conf = {"using_phone": 0.50, "no_seatbelt": 0.55}
    rejected_evidence: Dict[str, int] = {}
    confirmed_events: set[str] = set()
    max_conf = 0.0
    frame_idx = 0
    sampled_frames = 0
    best_frame = None
    best_conf = 0.0

    for source_frame_index in probe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_index))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1
        hierarchy = run_hierarchical_dms(
            frame,
            dms_detector=detector,
            pose_estimator=pose_estimator,
            vehicle_detector=vehicle_detector,
            config=HIERARCHY_CONFIG,
        )
        evidence = filter_dms_evidence(
            hierarchy.detections,
            hierarchy.pose,
            hierarchy.driver_roi,
            hierarchy.chest_roi,
            frame.shape,
        )
        detections = evidence.accepted
        for reason, count in evidence.rejected.items():
            rejected_evidence[reason] = rejected_evidence.get(reason, 0) + count
        pose_res = hierarchy.pose
        sampled_frames += 1

        # Draw Dual-Engine Geometry (Chest ROI & Keypoints)
        if pose_res and pose_res.points:
            draw_dual_engine_geometry(frame, pose_res.points)

        has_violation = False
        current_violation_conf = 0.0
        sampled_event_conf: Dict[str, float] = {}
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            c_name = det.class_name.lower()
            conf = det.confidence

            event = _event_for_detection(c_name)
            if event is None:
                continue
            color = (0, 0, 239) if event == "using_phone" else ((0, 215, 255) if event == "no_seatbelt" else (34, 197, 94))

            if event in violation_events:
                has_violation = True
                current_violation_conf = max(current_violation_conf, conf)
                sampled_event_conf[event] = max(sampled_event_conf.get(event, 0.0), conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{c_name} {conf:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for event in violation_events:
            confidence = sampled_event_conf.get(event, 0.0)
            event_histories[event].append(confidence)
            if confidence > 0:
                detected_event_counts[event] += 1
                event_peak_conf[event] = max(event_peak_conf[event], confidence)
            votes = sum(value > 0 for value in event_histories[event])
            positive_values = [value for value in event_histories[event] if value > 0]
            mean_confidence = (
                sum(positive_values) / len(positive_values) if positive_values else 0.0
            )
            if votes >= min_hits[event] and mean_confidence >= min_mean_conf[event]:
                confirmed_events.add(event)

        if has_violation and (best_frame is None or current_violation_conf > best_conf):
            best_frame = frame.copy()
            best_conf = current_violation_conf
        elif best_frame is None and frame_idx == 1:
            best_frame = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    primary_event = "normal"
    if confirmed_events:
        primary_event = max(confirmed_events, key=lambda event: detected_event_counts[event])
        max_conf = event_peak_conf[primary_event]

    # Save best frame image for web UI preview
    frames_dir = Path("outputs/cloud_uploads/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_filename = f"auto_detect_vframe_{timestamp_str}.jpg"
    frame_path = frames_dir / frame_filename
    if best_frame is not None:
        cv2.imwrite(str(frame_path), best_frame)

    return {
        "event_type": primary_event,
        "confidence": round(max_conf, 3),
        "total_frames": frame_idx,
        "source_total_frames": source_total_frames,
        "sampled_frames": sampled_frames,
        "sampling_strategy": "uniform_anchors_plus_scene_change_probes",
        "temporal_window_frames": 12,
        "confirmed_events": sorted(confirmed_events),
        "event_frame_votes": detected_event_counts,
        "rejected_evidence": rejected_evidence,
        "clip_path": str(out_path).replace("\\", "/"),
        "clip_url": f"/uploads/clips/{out_filename}",
        "frame_path": str(frame_path).replace("\\", "/") if best_frame is not None else None,
        "frame_url": f"/uploads/frames/{frame_filename}" if best_frame is not None else None,
    }
