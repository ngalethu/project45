from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.common.types import Detection, PoseResult
from app.common.utils import bbox_center, euclidean
from app.edge.behavior_rules import box_area, get_shoulder_width, overlap_ratio, point_in_box


@dataclass
class EvidenceResult:
    accepted: list[Detection] = field(default_factory=list)
    event_confidence: dict[str, float] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)


def _canonical(name: str) -> str:
    value = str(name).lower().strip().replace("_", "-").replace(" ", "-")
    aliases = {"cell-phone": "phone", "mobile-phone": "phone", "seat-belt": "seatbelt"}
    return aliases.get(value, value)


def _reject(result: EvidenceResult, reason: str) -> None:
    result.rejected[reason] = result.rejected.get(reason, 0) + 1


def _plausible_phone(
    detection: Detection,
    pose: PoseResult,
    driver_roi,
    frame_shape,
) -> tuple[bool, str]:
    height, width = frame_shape[:2]
    frame_area = max(1.0, float(width * height))
    area_ratio = box_area(detection.bbox) / frame_area
    box_width = max(1, detection.bbox[2] - detection.bbox[0])
    box_height = max(1, detection.bbox[3] - detection.bbox[1])
    aspect = max(box_width / box_height, box_height / box_width)

    if detection.confidence < 0.42:
        return False, "phone_low_confidence"
    if area_ratio < 0.00002 or area_ratio > 0.08 or aspect > 5.0:
        return False, "phone_implausible_geometry"
    if not pose.points:
        return False, "phone_missing_pose"

    center = bbox_center(detection.bbox)
    if driver_roi is not None and not point_in_box(center, driver_roi):
        return False, "phone_outside_driver_roi"

    shoulders = [pose.points.get("left_shoulder"), pose.points.get("right_shoulder")]
    wrists = [pose.points.get("left_wrist"), pose.points.get("right_wrist")]
    head = [pose.points.get("nose"), pose.points.get("left_ear"), pose.points.get("right_ear")]
    shoulders = [point for point in shoulders if point is not None]
    wrists = [point for point in wrists if point is not None]
    head = [point for point in head if point is not None]
    if not shoulders or not wrists or not head:
        return False, "phone_incomplete_pose"

    scale = get_shoulder_width(pose)
    hand_distance = min(euclidean(center, point) for point in wrists) / scale
    head_distance = min(euclidean(center, point) for point in head) / scale
    if hand_distance > 1.35 or head_distance > 2.25:
        return False, "phone_not_associated_with_driver"
    return True, ""


def filter_dms_evidence(
    detections: Iterable[Detection],
    pose: PoseResult,
    driver_roi,
    chest_roi,
    frame_shape,
) -> EvidenceResult:
    """Convert object boxes into conservative, spatially supported DMS events."""
    result = EvidenceResult()
    detections = list(detections)
    seatbelt = [det for det in detections if _canonical(det.class_name) == "seatbelt"]
    seatbelt_score = max((det.confidence for det in seatbelt), default=0.0)

    for detection in detections:
        name = _canonical(detection.class_name)
        if name == "phone":
            valid, reason = _plausible_phone(detection, pose, driver_roi, frame_shape)
            if not valid:
                _reject(result, reason)
                continue
            result.accepted.append(detection)
            result.event_confidence["using_phone"] = max(
                result.event_confidence.get("using_phone", 0.0), detection.confidence
            )
            continue

        if name == "seatbelt":
            result.accepted.append(detection)
            continue

        if name != "no-seatbelt":
            continue
        if detection.confidence < 0.50:
            _reject(result, "no_seatbelt_low_confidence")
            continue

        if chest_roi is not None:
            center_supported = point_in_box(bbox_center(detection.bbox), chest_roi)
            overlap_supported = overlap_ratio(detection.bbox, chest_roi) >= 0.35
            if not (center_supported or overlap_supported):
                _reject(result, "no_seatbelt_outside_chest_roi")
                continue
        elif detection.confidence < 0.72:
            _reject(result, "no_seatbelt_missing_chest_roi")
            continue

        if seatbelt_score and detection.confidence <= seatbelt_score + 0.10:
            _reject(result, "no_seatbelt_conflicts_with_seatbelt")
            continue

        result.accepted.append(detection)
        result.event_confidence["no_seatbelt"] = max(
            result.event_confidence.get("no_seatbelt", 0.0), detection.confidence
        )

    return result
