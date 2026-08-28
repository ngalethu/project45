"""Train, evaluate and export the four-class YOLO11 DMS detector.

Designed for both local runs and Kaggle GPU notebooks.  A run is only reported
as meeting the thesis target when *test-set* mAP@50 and macro F1 are both at
least 0.85; training completion alone is never presented as proof of accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
LOCAL_DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dms_yolo_4class_v2" / "dms_dataset.yaml"
LOCAL_OUTPUT_DIR = BACKEND_DIR / "outputs" / "runs_dms"
EXPECTED_NAMES = ["phone", "seatbelt", "no-seatbelt", "smoking"]


def configure_ultralytics_settings(settings: Any) -> dict[str, bool]:
    """Disable optional integrations without failing on older settings schemas."""
    requested = {"wandb": False, "raytune": False, "sync": False}
    available = set(settings.keys())
    supported = {key: value for key, value in requested.items() if key in available}
    if supported:
        settings.update(supported)
    skipped = sorted(set(requested) - set(supported))
    print(f"Ultralytics integrations disabled: {sorted(supported)}")
    if skipped:
        print(f"Ultralytics settings unavailable (safe skip): {skipped}")
    return supported


def _normalise_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    raise ValueError("Dataset YAML 'names' must be a list or mapping")


def validate_dataset_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = _normalise_names(data.get("names"))
    if names != EXPECTED_NAMES:
        raise ValueError(f"Expected class order {EXPECTED_NAMES}, got {names} in {yaml_path}")
    if int(data.get("nc", len(names))) != len(EXPECTED_NAMES):
        raise ValueError(f"Expected nc=4 in {yaml_path}")
    for split in ("train", "val", "test"):
        if split not in data:
            raise ValueError(f"Dataset YAML is missing '{split}': {yaml_path}")
    return data


def get_dataset_yaml_path(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.getenv("DMS_DATASET_YAML"):
        candidates.append(Path(os.environ["DMS_DATASET_YAML"]))
    candidates.append(LOCAL_DATASET_YAML)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.glob("*/dms_dataset.yaml")))
        candidates.extend(sorted(kaggle_input.glob("*/*/dms_dataset.yaml")))
    errors: list[str] = []
    for candidate in candidates:
        try:
            validate_dataset_yaml(candidate)
            return candidate.resolve()
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    raise FileNotFoundError("No valid four-class DMS dataset YAML found. " + " | ".join(errors))


def make_runtime_yaml(dataset_yaml: Path, run_dir: Path) -> Path:
    """Resolve a portable ``path: .`` relative to the YAML, not the process cwd."""
    data = validate_dataset_yaml(dataset_yaml)
    root = Path(data.get("path") or ".")
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    data["path"] = root.as_posix()
    runtime_yaml = run_dir / "runtime_dms_dataset.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_yaml


def _as_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (float, int)):
        return [float(value)]
    return [float(item) for item in value]


def summarise_metrics(metrics: Any) -> dict:
    box = metrics.box
    precision = float(getattr(box, "mp", 0.0))
    recall = float(getattr(box, "mr", 0.0))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    per_p = _as_list(getattr(box, "p", None))
    per_r = _as_list(getattr(box, "r", None))
    per_ap50 = _as_list(getattr(box, "ap50", None))
    per_map = _as_list(getattr(box, "ap", None))
    names = getattr(metrics, "names", {index: name for index, name in enumerate(EXPECTED_NAMES)})
    per_class: dict[str, dict[str, float]] = {}
    for index, name in sorted(names.items(), key=lambda item: int(item[0])):
        idx = int(index)
        p = per_p[idx] if idx < len(per_p) else 0.0
        r = per_r[idx] if idx < len(per_r) else 0.0
        per_class[str(name)] = {
            "precision": p,
            "recall": r,
            "f1": 2.0 * p * r / max(p + r, 1e-12),
            "map50": per_ap50[idx] if idx < len(per_ap50) else 0.0,
            "map50_95": per_map[idx] if idx < len(per_map) else 0.0,
        }
    summary = {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "target": {"map50": 0.85, "f1_macro": 0.85},
        "per_class": per_class,
    }
    summary["target_met"] = summary["map50"] >= 0.85 and summary["f1_macro"] >= 0.85
    return summary


def train_dms_model(
    dataset_yaml: str | Path | None = None,
    epochs: int = 120,
    batch_size: int = -1,
    img_size: int = 768,
    model_weights: str = "yolo11m.pt",
    output_dir: str | Path | None = None,
    run_name: str = "yolo11m_dms_4class_768",
    device: str | None = None,
    workers: int = 4,
    patience: int = 25,
    seed: int = 42,
    resume: str | Path | None = None,
    cache: bool | str = False,
    export_onnx: bool = True,
) -> dict:
    import torch
    from ultralytics import YOLO, settings

    os.environ.setdefault("WANDB_DISABLED", "true")
    configure_ultralytics_settings(settings)
    selected_yaml = get_dataset_yaml_path(dataset_yaml)
    output_root = Path(output_dir) if output_dir else (Path("/kaggle/working/runs_dms") if Path("/kaggle/working").exists() else LOCAL_OUTPUT_DIR)
    output_root = output_root.resolve()
    run_dir = output_root / run_name
    runtime_yaml = make_runtime_yaml(selected_yaml, run_dir)
    selected_device = device or ("0" if torch.cuda.is_available() else "cpu")
    starting_weights = str(resume or model_weights)

    print("=" * 78)
    print("YOLO11 DMS FOUR-CLASS TRAINING")
    print(f"dataset={selected_yaml}")
    print(f"model={starting_weights} device={selected_device} epochs={epochs} imgsz={img_size} batch={batch_size}")
    print(f"output={run_dir}")
    print("=" * 78)

    model = YOLO(starting_weights)
    train_kwargs = {
        "data": str(runtime_yaml),
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "device": selected_device,
        "workers": workers,
        "amp": True,
        "patience": patience,
        "seed": seed,
        "deterministic": True,
        "project": str(output_root),
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": 5,
        "plots": True,
        "cache": cache,
        "optimizer": "auto",
        "cos_lr": True,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.55,
        "hsv_v": 0.35,
        "degrees": 4.0,
        "translate": 0.08,
        "scale": 0.45,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.05,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
    }
    if resume:
        train_kwargs["resume"] = True
    training_result = model.train(**train_kwargs)

    trainer = getattr(model, "trainer", None)
    best_path = Path(getattr(trainer, "best", run_dir / "weights" / "best.pt"))
    if not best_path.exists():
        candidate = Path(getattr(training_result, "save_dir", run_dir)) / "weights" / "best.pt"
        best_path = candidate if candidate.exists() else best_path
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found under {run_dir}")

    best_model = YOLO(str(best_path))
    test_metrics = best_model.val(
        data=str(runtime_yaml),
        split="test",
        imgsz=img_size,
        batch=batch_size,
        device=selected_device,
        workers=workers,
        plots=True,
        project=str(output_root),
        name=f"{run_name}_test",
        exist_ok=True,
    )
    summary = summarise_metrics(test_metrics)
    summary.update(
        {
            "dataset_yaml": str(selected_yaml),
            "best_weights": str(best_path.resolve()),
            "model": model_weights,
            "epochs_requested": epochs,
            "image_size": img_size,
            "seed": seed,
        }
    )

    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, artifact_dir / "best.pt")
    if export_onnx:
        exported = best_model.export(format="onnx", imgsz=img_size, simplify=True, dynamic=True)
        exported_path = Path(str(exported))
        if exported_path.exists():
            shutil.copy2(exported_path, artifact_dir / "best.onnx")
            summary["onnx"] = str((artifact_dir / "best.onnx").resolve())
    metrics_path = artifact_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Metrics: {metrics_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name", default="yolo11m_dms_4class_768")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--no-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache: bool | str = False if args.cache == "false" else args.cache
    train_dms_model(
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        model_weights=args.model,
        output_dir=args.output_dir,
        run_name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        cache=cache,
        export_onnx=not args.no_onnx,
    )


if __name__ == "__main__":
    main()
