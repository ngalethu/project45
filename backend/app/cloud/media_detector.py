from __future__ import annotations
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from app.edge.yolo_detector import YoloDetector
from app.edge.pose_estimator import PoseEstimator

_DETECTOR = None
_POSE_ESTIMATOR = None

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

    detector = get_yolo_detector()
    pose_estimator = get_pose_estimator()

    # 1. Trích xuất 33 điểm mốc tư thế cơ thể (MediaPipe Pose Keypoints)
    pose_res = pose_estimator.predict(frame)
    has_chest_roi = draw_dual_engine_geometry(frame, pose_res.points)

    # 2. YOLO11 Bounding Box Detection
    detections = detector.predict(frame)

    detected_events = []
    max_conf = 0.85
    primary_event = "normal"

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        c_name = det.class_name.lower()
        conf = det.confidence

        if "phone" in c_name:
            event = "using_phone"
            color = (0, 0, 239)
        elif "no-seatbelt" in c_name or "no_seatbelt" in c_name:
            event = "no_seatbelt"
            color = (0, 215, 255)
        else:
            event = "seatbelt"
            color = (34, 197, 94)

        if event in ["using_phone", "no_seatbelt"]:
            detected_events.append(event)
            if conf > max_conf or primary_event == "normal":
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
        "frame_path": str(out_path).replace("\\", "/"),
        "frame_url": f"/uploads/frames/{out_filename}",
    }

def process_uploaded_video(file_path: str, max_frames: int = 150) -> Dict[str, Any]:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Invalid video file format")

    detector = get_yolo_detector()
    pose_estimator = get_pose_estimator()

    timestamp_str = int(time.time() * 1000)
    out_dir = Path("outputs/cloud_uploads/clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"auto_detect_{timestamp_str}.mp4"
    out_path = out_dir / out_filename

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    out = None
    codecs_to_try = [("avc1", "mp4"), ("H264", "mp4"), ("mp4v", "mp4")]
    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if writer.isOpened():
                out = writer
                break
        except Exception:
            continue
            
    if out is None or not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    detected_event_counts: Dict[str, int] = {}
    max_conf = 0.80
    frame_idx = 0
    SAMPLE_INTERVAL = 4

    best_frame = None
    best_conf = 0.0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Only run YOLO model inference & MediaPipe Pose on sampled frames for high performance
        if frame_idx % SAMPLE_INTERVAL == 1:
            detections = detector.predict(frame)
            pose_res = pose_estimator.predict(frame)
            current_dets = detections
            current_pose = pose_res
        else:
            detections = current_dets if 'current_dets' in locals() else []
            pose_res = current_pose if 'current_pose' in locals() else None

        # Draw Dual-Engine Geometry (Chest ROI & Keypoints)
        if pose_res and pose_res.points:
            draw_dual_engine_geometry(frame, pose_res.points)

        has_violation = False
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            c_name = det.class_name.lower()
            conf = det.confidence

            if "phone" in c_name:
                event = "using_phone"
                color = (0, 0, 239)
            elif "no-seatbelt" in c_name or "no_seatbelt" in c_name:
                event = "no_seatbelt"
                color = (0, 215, 255)
            else:
                event = "seatbelt"
                color = (34, 197, 94)

            if event in ["using_phone", "no_seatbelt"]:
                detected_event_counts[event] = detected_event_counts.get(event, 0) + 1
                has_violation = True
                if conf > max_conf:
                    max_conf = conf

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{c_name} {conf:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if has_violation and (best_frame is None or max_conf > best_conf):
            best_frame = frame.copy()
            best_conf = max_conf
        elif best_frame is None and frame_idx == 1:
            best_frame = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    primary_event = "normal"
    if detected_event_counts:
        primary_event = max(detected_event_counts, key=detected_event_counts.get)

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
        "clip_path": str(out_path).replace("\\", "/"),
        "clip_url": f"/uploads/clips/{out_filename}",
        "frame_path": str(frame_path).replace("\\", "/") if best_frame is not None else None,
        "frame_url": f"/uploads/frames/{frame_filename}" if best_frame is not None else None,
    }
