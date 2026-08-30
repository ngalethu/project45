"""Evaluate each sample video as one independent multi-label event sample."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.cloud.media_detector import process_uploaded_video  # noqa: E402


SAMPLE_VIDEOS_DIR = PROJECT_ROOT / "data" / "sample_videos"
DEFAULT_MANIFEST = SAMPLE_VIDEOS_DIR / "benchmark_manifest.json"
OUTPUT_PATH = BACKEND_DIR / "outputs" / "benchmarks" / "video_evaluation_metrics.json"
EVENTS = ("using_phone", "no_seatbelt", "normal")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    precision = _safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        **counts,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _predicted_events(result: dict) -> set[str]:
    violations = {
        event for event in result.get("confirmed_events", [])
        if event in {"using_phone", "no_seatbelt"}
    }
    return violations or {"normal"}


def evaluate_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("videos") or []
    if not samples:
        raise ValueError(f"No videos declared in {manifest_path}")

    counts = {event: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for event in EVENTS}
    details = []
    started = time.time()
    for sample in samples:
        video_path = SAMPLE_VIDEOS_DIR / sample["file"]
        if not video_path.is_file():
            details.append({**sample, "status": "missing"})
            continue
        expected = set(sample.get("events") or ["normal"])
        before = time.time()
        try:
            result = process_uploaded_video(str(video_path))
            predicted = _predicted_events(result)
            for event in EVENTS:
                expected_positive = event in expected
                predicted_positive = event in predicted
                if expected_positive and predicted_positive:
                    counts[event]["tp"] += 1
                elif not expected_positive and predicted_positive:
                    counts[event]["fp"] += 1
                elif expected_positive and not predicted_positive:
                    counts[event]["fn"] += 1
                else:
                    counts[event]["tn"] += 1
            details.append({
                **sample,
                "status": "evaluated",
                "expected_events": sorted(expected),
                "predicted_events": sorted(predicted),
                "exact_match": expected == predicted,
                "confidence": result.get("confidence", 0.0),
                "sampled_frames": result.get("sampled_frames", 0),
                "event_frame_votes": result.get("event_frame_votes", {}),
                "rejected_evidence": result.get("rejected_evidence", {}),
                "elapsed_sec": round(time.time() - before, 3),
            })
        except Exception as exc:
            details.append({**sample, "status": "error", "error": str(exc)})

    per_event = {event: _metrics(event_counts) for event, event_counts in counts.items()}
    evaluated = [item for item in details if item.get("status") == "evaluated"]
    exact_matches = sum(bool(item["exact_match"]) for item in evaluated)
    violation_events = ("using_phone", "no_seatbelt")
    micro_counts = {
        key: sum(counts[event][key] for event in violation_events)
        for key in ("tp", "fp", "fn", "tn")
    }
    payload = {
        "schema_version": 2,
        "evaluation_unit": "one video contributes one decision per event",
        "manifest": str(manifest_path.resolve()),
        "videos_declared": len(samples),
        "videos_evaluated": len(evaluated),
        "exact_match_accuracy": round(_safe_ratio(exact_matches, len(evaluated)), 4),
        "per_event": per_event,
        "violation_micro": _metrics(micro_counts),
        "violation_macro_f1": round(
            sum(per_event[event]["f1"] for event in violation_events) / len(violation_events), 4
        ),
        "elapsed_sec": round(time.time() - started, 3),
        "details": details,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    evaluate_manifest()
