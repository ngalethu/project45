from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.common.types import Detection, PoseResult
from app.edge.behavior_rules import build_chest_roi, build_driver_roi


@dataclass
class HierarchicalResult:
    detections: list[Detection]
    pose: PoseResult
    vehicle_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    windshield_rois: list[tuple[int, int, int, int]] = field(default_factory=list)
    driver_roi: tuple[int, int, int, int] | None = None
    chest_roi: tuple[int, int, int, int] | None = None
    stages: list[str] = field(default_factory=list)


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def windshield_from_vehicle(
    vehicle_box: tuple[int, int, int, int],
    frame_shape,
    x_margin: float = 0.12,
    y_start: float = 0.04,
    y_end: float = 0.58,
) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = vehicle_box
    vehicle_w, vehicle_h = x2 - x1, y2 - y1
    return _clip_box(
        (
            x1 + vehicle_w * x_margin,
            y1 + vehicle_h * y_start,
            x2 - vehicle_w * x_margin,
            y1 + vehicle_h * y_end,
        ),
        width,
        height,
    )


def _offset_detections(
    detections: list[Detection], roi: tuple[int, int, int, int]
) -> list[Detection]:
    offset_x, offset_y = roi[0], roi[1]
    return [
        Detection(
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=(
                det.bbox[0] + offset_x,
                det.bbox[1] + offset_y,
                det.bbox[2] + offset_x,
                det.bbox[3] + offset_y,
            ),
        )
        for det in detections
    ]


def _offset_pose(pose: PoseResult, roi: tuple[int, int, int, int]) -> PoseResult:
    return PoseResult(points={
        name: (point[0] + roi[0], point[1] + roi[1])
        for name, point in pose.points.items()
    })


def _iou(left, right) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def merge_detections(detections: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            current.class_name == detection.class_name
            and _iou(current.bbox, detection.bbox) >= iou_threshold
            for current in kept
        ):
            continue
        kept.append(detection)
    return kept


def run_hierarchical_dms(
    frame,
    dms_detector: Any,
    pose_estimator: Any | None = None,
    vehicle_detector: Any | None = None,
    config: dict | None = None,
) -> HierarchicalResult:
    cfg = config or {}
    height, width = frame.shape[:2]
    dms_imgsz = int(cfg.get("dms_imgsz", 768))
    chest_imgsz = int(cfg.get("chest_imgsz", dms_imgsz))
    max_vehicles = max(1, int(cfg.get("max_vehicles", 3)))
    min_vehicle_area = float(cfg.get("min_vehicle_area_ratio", 0.025)) * width * height
    vehicle_names = {str(name).lower() for name in cfg.get("vehicle_names", ["car", "truck", "bus"])}

    vehicle_boxes: list[tuple[int, int, int, int]] = []
    windshield_rois: list[tuple[int, int, int, int]] = []
    stages: list[str] = []
    if vehicle_detector is not None and cfg.get("vehicle_stage_enabled", True):
        vehicle_detections = vehicle_detector.predict(frame, imgsz=int(cfg.get("vehicle_imgsz", 640)))
        candidates = [
            detection
            for detection in vehicle_detections
            if detection.class_name in vehicle_names
            and (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]) >= min_vehicle_area
        ]
        candidates.sort(
            key=lambda detection: (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]),
            reverse=True,
        )
        vehicle_boxes = [detection.bbox for detection in candidates[:max_vehicles]]
        windshield_rois = [
            roi
            for roi in (
                windshield_from_vehicle(
                    box,
                    frame.shape,
                    x_margin=float(cfg.get("windshield_x_margin", 0.12)),
                    y_start=float(cfg.get("windshield_y_start", 0.04)),
                    y_end=float(cfg.get("windshield_y_end", 0.58)),
                )
                for box in vehicle_boxes
            )
            if roi is not None
        ]
        if windshield_rois:
            stages.extend(["vehicle", "windshield"])

    primary_roi = windshield_rois[0] if windshield_rois else (0, 0, width, height)
    primary_crop = frame[primary_roi[1] : primary_roi[3], primary_roi[0] : primary_roi[2]]
    pose = PoseResult(points={})
    if pose_estimator is not None:
        local_pose = pose_estimator.predict(primary_crop)
        pose = _offset_pose(local_pose, primary_roi)
        if pose.points:
            stages.append("pose")

    driver_roi = build_driver_roi(pose, width, height) if pose.points else None
    driver_roi = _clip_box(driver_roi, width, height) if driver_roi else None
    if driver_roi:
        stages.append("driver_roi")

    all_detections: list[Detection] = []
    inference_rois = windshield_rois or [primary_roi]
    for index, roi in enumerate(inference_rois):
        effective_roi = driver_roi if index == 0 and driver_roi else roi
        crop = frame[effective_roi[1] : effective_roi[3], effective_roi[0] : effective_roi[2]]
        if crop.size:
            all_detections.extend(_offset_detections(dms_detector.predict(crop, imgsz=dms_imgsz), effective_roi))
    stages.append("dms")

    chest_roi = build_chest_roi(pose, width, height) if pose.points else None
    chest_roi = _clip_box(chest_roi, width, height) if chest_roi else None
    if chest_roi and cfg.get("chest_second_pass_enabled", True):
        chest_crop = frame[chest_roi[1] : chest_roi[3], chest_roi[0] : chest_roi[2]]
        if chest_crop.size:
            chest_detections = _offset_detections(
                dms_detector.predict(chest_crop, imgsz=chest_imgsz), chest_roi
            )
            allowed = {"seatbelt", "no-seatbelt", "no_seatbelt", "no seatbelt"}
            all_detections.extend(
                detection for detection in chest_detections if detection.class_name in allowed
            )
            stages.append("chest_second_pass")

    return HierarchicalResult(
        detections=merge_detections(all_detections),
        pose=pose,
        vehicle_boxes=vehicle_boxes,
        windshield_rois=windshield_rois,
        driver_roi=driver_roi,
        chest_roi=chest_roi,
        stages=stages,
    )
