"""Create conservative YOLO pseudo-labels from weak DMS sources.

AUC Distracted Driver is an image-classification dataset and Seatbelt Real has
no bounding boxes in the downloaded archive.  This script deliberately keeps
only high-confidence boxes predicted by an already trained four-class teacher;
it never converts a whole-image class into a fake object box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

import cv2
import numpy as np
import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_AUC_ZIP = PROJECT_ROOT / "data" / "sources" / "_archives" / "auc.distracted.driver.dataset_v2.zip"
DEFAULT_SEATBELT_ZIP = Path.home() / "Downloads" / "archive (1).zip"
DEFAULT_BASE_YAML = PROJECT_ROOT / "data" / "processed" / "dms_yolo_3class_v3_curated" / "dms_dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dms_weak_pseudolabels_v1"
CANONICAL_NAMES = ["phone", "seatbelt", "no-seatbelt"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUC_PHONE_CLASSES = {"c1", "c2", "c3", "c4"}


def canon_name(value: str) -> str:
    value = str(value).lower().replace("_", "-").strip()
    aliases = {
        "mobile": "phone",
        "mobile-phone": "phone",
        "cell-phone": "phone",
        "seat-belt": "seatbelt",
        "no seatbelt": "no-seatbelt",
    }
    return aliases.get(value, value)


def auc_phone_entries(archive: zipfile.ZipFile) -> list[str]:
    """Use v2 only; v1 and v2 contain overlapping AUC material."""
    selected: list[str] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = [part.lower() for part in path.parts]
        if not parts or not parts[0].startswith("v2_cam1_cam2"):
            continue
        if any(part in AUC_PHONE_CLASSES for part in parts):
            selected.append(info.filename)
    return selected


def image_entries(archive: zipfile.ZipFile) -> list[str]:
    return [
        info.filename
        for info in archive.infolist()
        if PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() in IMAGE_SUFFIXES
    ]


def inspect_archives(auc_zip: Path, seatbelt_zip: Path) -> dict:
    report: dict = {}
    if auc_zip.exists():
        with zipfile.ZipFile(auc_zip) as archive:
            report["auc_phone_candidates"] = len(auc_phone_entries(archive))
            report["auc_entries"] = len(archive.infolist())
            report["auc_encrypted_entries"] = sum(bool(info.flag_bits & 0x1) for info in archive.infolist())
    else:
        report["auc_missing"] = str(auc_zip)
    if seatbelt_zip.exists():
        with zipfile.ZipFile(seatbelt_zip) as archive:
            report["seatbelt_candidates"] = len(image_entries(archive))
    else:
        report["seatbelt_missing"] = str(seatbelt_zip)
    return report


def _model_class_ids(model) -> dict[str, int]:
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    mapped = {canon_name(name): int(index) for index, name in names.items()}
    missing = [name for name in CANONICAL_NAMES if name not in mapped]
    if missing:
        raise ValueError(f"Teacher model is missing canonical classes: {missing}; names={names}")
    return mapped


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _safe_output_name(source: str, entry_name: str) -> str:
    path = PurePosixPath(entry_name.replace("\\", "/"))
    digest = hashlib.sha1(entry_name.encode("utf-8")).hexdigest()[:12]
    return f"{source}__{path.stem}__{digest}{path.suffix.lower()}"


def _pseudo_label_archive(
    model,
    archive_path: Path,
    entries: list[str],
    source_name: str,
    allowed_class: str,
    class_ids: dict[str, int],
    output_dir: Path,
    confidence: float,
    image_size: int,
    batch_size: int,
    device: str | None,
    limit: int | None,
    password: str | None = None,
) -> Counter:
    stats = Counter()
    allowed_id = class_ids[allowed_class]
    selected = entries[:limit] if limit else entries
    with zipfile.ZipFile(archive_path) as archive:
        for batch_names in _chunks(selected, batch_size):
            images: list[np.ndarray] = []
            encoded: list[bytes] = []
            valid_names: list[str] = []
            for entry_name in batch_names:
                try:
                    payload = archive.read(entry_name, pwd=password.encode("utf-8") if password else None)
                    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                except (KeyError, OSError, RuntimeError, zipfile.BadZipFile):
                    image = None
                    payload = b""
                if image is None:
                    stats["unreadable"] += 1
                    continue
                images.append(image)
                encoded.append(payload)
                valid_names.append(entry_name)
            if not images:
                continue
            predictions = model.predict(
                source=images,
                imgsz=image_size,
                conf=confidence,
                device=device,
                classes=[allowed_id],
                verbose=False,
            )
            for entry_name, payload, result in zip(valid_names, encoded, predictions):
                stats["scanned"] += 1
                lines: list[str] = []
                if result.boxes is not None:
                    classes = result.boxes.cls.detach().cpu().tolist()
                    boxes = result.boxes.xywhn.detach().cpu().tolist()
                    confidences = result.boxes.conf.detach().cpu().tolist()
                    for predicted_id, box, score in zip(classes, boxes, confidences):
                        if int(predicted_id) != allowed_id or float(score) < confidence:
                            continue
                        canonical_id = CANONICAL_NAMES.index(allowed_class)
                        lines.append(f"{canonical_id} " + " ".join(f"{float(value):.8g}" for value in box))
                        stats["boxes"] += 1
                if not lines:
                    stats["rejected_no_box"] += 1
                    continue
                output_name = _safe_output_name(source_name, entry_name)
                image_path = output_dir / "images" / "train" / output_name
                label_path = output_dir / "labels" / "train" / f"{Path(output_name).stem}.txt"
                image_path.write_bytes(payload)
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                stats["accepted_images"] += 1
    return stats


def create_combined_yaml(base_yaml: Path, pseudo_dir: Path) -> Path:
    data = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    ordered_names = list(names.values()) if isinstance(names, dict) else list(names or [])
    if ordered_names != CANONICAL_NAMES:
        raise ValueError(f"Base dataset must use {CANONICAL_NAMES}; got {ordered_names}")
    base_root = Path(data.get("path") or ".")
    if not base_root.is_absolute():
        base_root = (base_yaml.parent / base_root).resolve()
    combined = {
        "path": "/",
        "train": [
            (base_root / data["train"]).resolve().as_posix(),
            (pseudo_dir / "images" / "train").resolve().as_posix(),
        ],
        "val": (base_root / data["val"]).resolve().as_posix(),
        "test": (base_root / data["test"]).resolve().as_posix(),
        "nc": len(CANONICAL_NAMES),
        "names": dict(enumerate(CANONICAL_NAMES)),
    }
    output = pseudo_dir / "dms_dataset_with_pseudo.yaml"
    output.write_text(yaml.safe_dump(combined, sort_keys=False), encoding="utf-8")
    return output


def generate_pseudo_labels(
    weights: Path,
    auc_zip: Path = DEFAULT_AUC_ZIP,
    seatbelt_zip: Path = DEFAULT_SEATBELT_ZIP,
    base_yaml: Path = DEFAULT_BASE_YAML,
    output_dir: Path = DEFAULT_OUTPUT,
    confidence: float = 0.70,
    image_size: int = 768,
    batch_size: int = 16,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    auc_password: str | None = None,
) -> dict:
    from ultralytics import YOLO

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output_dir}; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    class_ids = _model_class_ids(model)
    report = {"confidence": confidence, "teacher": str(weights.resolve()), "sources": {}}
    if auc_zip.exists():
        with zipfile.ZipFile(auc_zip) as archive:
            entries = auc_phone_entries(archive)
            encrypted = any(info.flag_bits & 0x1 for info in archive.infolist())
        if encrypted and not auc_password:
            report["sources"]["auc_phone"] = {
                "blocked_encrypted_archive": True,
                "phone_candidates": len(entries),
                "accepted_images": 0,
            }
        else:
            report["sources"]["auc_phone"] = dict(
                _pseudo_label_archive(
                    model, auc_zip, entries, "auc_phone", "phone", class_ids, output_dir,
                    confidence, image_size, batch_size, device, limit, auc_password,
                )
            )
    if seatbelt_zip.exists():
        with zipfile.ZipFile(seatbelt_zip) as archive:
            entries = image_entries(archive)
        report["sources"]["seatbelt_real"] = dict(
            _pseudo_label_archive(
                model, seatbelt_zip, entries, "seatbelt_real", "seatbelt", class_ids, output_dir,
                confidence, image_size, batch_size, device, limit, None,
            )
        )

    combined_yaml = create_combined_yaml(base_yaml, output_dir)
    report["combined_yaml"] = str(combined_yaml.resolve())
    (output_dir / "pseudo_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--auc-zip", type=Path, default=DEFAULT_AUC_ZIP)
    parser.add_argument("--seatbelt-zip", type=Path, default=DEFAULT_SEATBELT_ZIP)
    parser.add_argument("--base-yaml", type=Path, default=DEFAULT_BASE_YAML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auc-password", default=os.getenv("AUC_ZIP_PASSWORD"))
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect:
        print(json.dumps(inspect_archives(args.auc_zip, args.seatbelt_zip), indent=2))
        return
    if args.weights is None:
        raise SystemExit("--weights is required unless --inspect is used")
    generate_pseudo_labels(
        weights=args.weights,
        auc_zip=args.auc_zip,
        seatbelt_zip=args.seatbelt_zip,
        base_yaml=args.base_yaml,
        output_dir=args.output_dir,
        confidence=args.conf,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        auc_password=args.auc_password,
    )


if __name__ == "__main__":
    main()
