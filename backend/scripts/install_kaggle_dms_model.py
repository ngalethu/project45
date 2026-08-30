"""Validate and atomically install Kaggle DMS artifacts into backend/models."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BACKEND_DIR / "models"
EXPECTED_NAMES = ["phone", "seatbelt", "no-seatbelt"]


def _normalise_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    return [str(value) for value in names]


def validate_artifacts(best_pt: Path, metrics_path: Path, allow_below_target: bool) -> dict:
    if not best_pt.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing best.pt or metrics file: {best_pt}, {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = list((metrics.get("per_class") or {}).keys())
    if per_class != EXPECTED_NAMES:
        raise ValueError(f"Metrics classes must be {EXPECTED_NAMES}; got {per_class}")
    if not allow_below_target and not metrics.get("target_met", False):
        raise ValueError("Kaggle metrics did not meet the configured mAP50/F1 target")

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    names = _normalise_names(model.names)
    if names != EXPECTED_NAMES:
        raise ValueError(f"Checkpoint classes must be {EXPECTED_NAMES}; got {names}")
    return metrics


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".installing")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def install(
    best_pt: Path,
    metrics_path: Path,
    best_onnx: Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    allow_below_target: bool = False,
) -> dict:
    best_pt = best_pt.resolve()
    metrics_path = metrics_path.resolve()
    best_onnx = best_onnx.resolve() if best_onnx else None
    if best_onnx and not best_onnx.is_file():
        raise FileNotFoundError(best_onnx)
    metrics = validate_artifacts(best_pt, metrics_path, allow_below_target)

    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = model_dir / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ("best.pt", "best.onnx", "metrics_summary.json", "deployment_manifest.json"):
        current = model_dir / name
        if current.exists():
            shutil.copy2(current, backup_dir / name)

    _atomic_copy(best_pt, model_dir / "best.pt")
    if best_onnx:
        _atomic_copy(best_onnx, model_dir / "best.onnx")
    _atomic_copy(metrics_path, model_dir / "metrics_summary.json")
    manifest = {
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "classes": EXPECTED_NAMES,
        "source_best_pt": str(best_pt),
        "source_best_onnx": str(best_onnx) if best_onnx else None,
        "source_metrics": str(metrics_path),
        "backup_dir": str(backup_dir),
        "metrics": {
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "f1_macro": metrics.get("f1_macro"),
            "target_met": metrics.get("target_met"),
        },
    }
    (model_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-pt", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--best-onnx", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--allow-below-target", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    install(
        best_pt=args.best_pt,
        metrics_path=args.metrics,
        best_onnx=args.best_onnx,
        model_dir=args.model_dir,
        allow_below_target=args.allow_below_target,
    )
