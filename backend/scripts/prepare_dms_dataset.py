"""Build a leakage-aware, four-class YOLO dataset for the DMS project.

The local ``data/raw`` directory contains files consolidated from multiple
datasets.  Class ids are source-specific, so applying one global mapping (the
old behaviour) silently corrupts labels.  This module identifies every source
from its filename prefix and applies the corresponding ontology:

0 phone, 1 seatbelt, 2 no-seatbelt, 3 smoking.

AUC Distracted Driver and the eight Seatbelt Real images are weakly labelled
sources: they are audited and recorded in a manifest but are not mixed into an
object-detection dataset until a teacher model creates bounding boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
import yaml


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "data"
RAW_DATASET_DIR = DATASET_ROOT / "raw"
OUTPUT_DATASET_DIR = DATASET_ROOT / "processed" / "dms_yolo_4class_v2"
AUC_ARCHIVE = DATASET_ROOT / "sources" / "_archives" / "auc.distracted.driver.dataset_v2.zip"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
CANONICAL_CLASSES = {
    0: "phone",
    1: "seatbelt",
    2: "no-seatbelt",
    3: "smoking",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    prefixes: tuple[str, ...]
    label_map: Mapping[int, int | None]
    class_names: Mapping[int, str]
    weak_only: bool = False


# Keep the most specific prefix first.
SOURCE_SPECS = (
    SourceSpec(
        name="roboflow_seatbelt_mobile",
        prefixes=("raw_roboflow_seatbelt_mobile_", "roboflow_seatbelt_mobile_"),
        label_map={0: 0, 1: 1, 2: None},
        class_names={0: "mobile", 1: "seatbelt", 2: "windshield"},
    ),
    SourceSpec(
        name="roboflow_v9_primary",
        prefixes=("raw_roboflow_",),
        label_map={0: 2, 1: 0, 2: 1, 3: 3},
        class_names={0: "no-seatbelt", 1: "phone", 2: "seatbelt", 3: "smoking"},
    ),
    SourceSpec(
        name="dms_safety",
        prefixes=("dms_safety_",),
        label_map={0: None, 1: None, 2: 3, 3: 0, 4: 1},
        class_names={0: "open-eye", 1: "closed-eye", 2: "cigarette", 3: "phone", 4: "seatbelt"},
    ),
    SourceSpec(
        name="seatbelt_real_unlabelled",
        prefixes=("real_car_",),
        label_map={},
        class_names={},
        weak_only=True,
    ),
)


@dataclass(frozen=True)
class ImageRecord:
    source: SourceSpec
    image_path: Path
    label_path: Path
    original_split: str
    output_split: str
    group_key: str


@dataclass
class LabelResult:
    lines: list[str]
    kept: Counter
    dropped: Counter
    invalid_lines: int = 0
    unknown_classes: int = 0


def source_for_filename(filename: str) -> SourceSpec | None:
    lower = filename.lower()
    for spec in SOURCE_SPECS:
        if any(lower.startswith(prefix) for prefix in spec.prefixes):
            return spec
    return None


def _without_source_prefix(filename: str, spec: SourceSpec) -> str:
    lower = filename.lower()
    for prefix in spec.prefixes:
        if lower.startswith(prefix):
            return filename[len(prefix) :]
    return filename


def group_key_for(filename: str, spec: SourceSpec) -> str:
    """Return a stable capture/augmentation group for leakage-safe splitting."""
    stem = Path(_without_source_prefix(filename, spec)).stem.lower()
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    if spec.name == "dms_safety":
        # Frames from one source video must stay in the same split.
        stem = re.split(r"_mp4[-_]", stem, maxsplit=1)[0]
    return f"{spec.name}:{stem}"


def deterministic_split(group_key: str, train: float = 0.80, val: float = 0.10) -> str:
    bucket = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def discover_records(raw_dir: Path, split_mode: str = "grouped") -> tuple[list[ImageRecord], list[Path]]:
    records: list[ImageRecord] = []
    unknown_images: list[Path] = []
    for split in SPLITS:
        image_dir = raw_dir / split / "images"
        label_dir = raw_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            spec = source_for_filename(image_path.name)
            if spec is None:
                unknown_images.append(image_path)
                continue
            group_key = group_key_for(image_path.name, spec)
            output_split = deterministic_split(group_key) if split_mode == "grouped" else split
            records.append(
                ImageRecord(
                    source=spec,
                    image_path=image_path,
                    label_path=label_dir / f"{image_path.stem}.txt",
                    original_split=split,
                    output_split=output_split,
                    group_key=group_key,
                )
            )
    return records, unknown_images


def harmonize_label(label_path: Path, spec: SourceSpec) -> LabelResult:
    result = LabelResult(lines=[], kept=Counter(), dropped=Counter())
    if not label_path.exists():
        return result

    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            result.invalid_lines += 1
            continue
        try:
            raw_class = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except (TypeError, ValueError):
            result.invalid_lines += 1
            continue
        if not all(math.isfinite(value) for value in coords):
            result.invalid_lines += 1
            continue
        x, y, width, height = coords
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            result.invalid_lines += 1
            continue
        if raw_class not in spec.label_map:
            result.unknown_classes += 1
            continue
        canonical = spec.label_map[raw_class]
        source_name = spec.class_names.get(raw_class, str(raw_class))
        if canonical is None:
            result.dropped[source_name] += 1
            continue
        result.kept[CANONICAL_CLASSES[canonical]] += 1
        result.lines.append(f"{canonical} " + " ".join(f"{value:.8g}" for value in coords))
    return result


def audit_dataset(raw_dir: Path = RAW_DATASET_DIR, split_mode: str = "grouped") -> dict:
    records, unknown_images = discover_records(raw_dir, split_mode=split_mode)
    report: dict = {
        "raw_dir": str(raw_dir.resolve()),
        "split_mode": split_mode,
        "images_total": len(records) + len(unknown_images),
        "detection_images": 0,
        "weak_images": 0,
        "unknown_images": len(unknown_images),
        "missing_labels": 0,
        "invalid_label_lines": 0,
        "unknown_class_lines": 0,
        "source_images": Counter(),
        "original_split_images": Counter(),
        "output_split_images": Counter(),
        "class_instances": Counter(),
        "dropped_instances": Counter(),
    }
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        report["source_images"][record.source.name] += 1
        report["original_split_images"][record.original_split] += 1
        group_splits[record.group_key].add(record.original_split)
        if record.source.weak_only:
            report["weak_images"] += 1
            continue
        report["detection_images"] += 1
        report["output_split_images"][record.output_split] += 1
        if not record.label_path.exists():
            report["missing_labels"] += 1
            continue
        labels = harmonize_label(record.label_path, record.source)
        report["invalid_label_lines"] += labels.invalid_lines
        report["unknown_class_lines"] += labels.unknown_classes
        report["class_instances"].update(labels.kept)
        report["dropped_instances"].update(labels.dropped)

    report["groups_crossing_original_splits"] = sum(len(splits) > 1 for splits in group_splits.values())
    report["auc_archive"] = {
        "path": str(AUC_ARCHIVE.resolve()),
        "exists": AUC_ARCHIVE.exists(),
        "annotation_type": "image-classification",
    }
    for key in (
        "source_images",
        "original_split_images",
        "output_split_images",
        "class_instances",
        "dropped_instances",
    ):
        report[key] = dict(sorted(report[key].items()))
    return report


def read_image_win(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image_win(path: Path, image: np.ndarray) -> bool:
    try:
        extension = path.suffix or ".jpg"
        ok, buffer = cv2.imencode(extension, image)
        if not ok:
            return False
        buffer.tofile(str(path))
        return True
    except Exception:
        return False


def apply_clahe_preprocessing(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(luminance)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _copy_image(
    source: Path,
    destination: Path,
    split: str,
    clahe_mode: str,
    dark_threshold: float,
    hardlink: bool,
) -> tuple[bool, bool]:
    """Copy/link one image and return (success, clahe_applied)."""
    apply_clahe = False
    image: np.ndarray | None = None
    if clahe_mode != "none" and split == "train":
        image = read_image_win(source)
        if image is None:
            return False, False
        if clahe_mode == "all":
            apply_clahe = True
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            apply_clahe = float(lab[:, :, 0].mean()) < dark_threshold
    if apply_clahe:
        return write_image_win(destination, apply_clahe_preprocessing(image)), True
    try:
        if hardlink:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
        return True, False
    except OSError:
        try:
            shutil.copy2(source, destination)
            return True, False
        except OSError:
            return False, False


def _portable_yaml(output_dir: Path) -> dict:
    return {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }


def _write_weak_manifest(output_dir: Path, weak_records: Iterable[ImageRecord]) -> None:
    payload = {
        "policy": "Do not use as YOLO labels until a teacher model creates bounding boxes.",
        "auc": {
            "archive": str(AUC_ARCHIVE.resolve()),
            "available": AUC_ARCHIVE.exists(),
            "type": "classification",
            "phone_positive_categories": ["Text Left", "Text Right", "Talk Left", "Talk Right", "c1", "c2", "c3", "c4"],
        },
        "seatbelt_real": {
            "type": "unlabelled_detection_hard_cases",
            "images": [str(record.image_path.resolve()) for record in weak_records],
        },
    }
    (output_dir / "weak_sources_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_dataset(
    raw_dir: Path = RAW_DATASET_DIR,
    output_dir: Path = OUTPUT_DATASET_DIR,
    split_mode: str = "grouped",
    clahe_mode: str = "dark-only",
    dark_threshold: float = 85.0,
    hardlink: bool = True,
    overwrite: bool = False,
    make_zip: bool = False,
) -> dict:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output_dir}. Use --overwrite to rebuild it.")
        if output_dir == raw_dir or raw_dir in output_dir.parents:
            raise ValueError("Refusing to remove an output directory inside the raw dataset")
        shutil.rmtree(output_dir)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records, unknown_images = discover_records(raw_dir, split_mode=split_mode)
    audit = audit_dataset(raw_dir, split_mode=split_mode)
    build_stats = Counter()
    build_failures: list[str] = []
    weak_records: list[ImageRecord] = []
    used_names: set[tuple[str, str]] = set()

    for index, record in enumerate(records, start=1):
        if record.source.weak_only:
            weak_records.append(record)
            continue
        if not record.label_path.exists():
            build_stats["skipped_missing_label"] += 1
            continue

        labels = harmonize_label(record.label_path, record.source)
        clean_name = _without_source_prefix(record.image_path.name, record.source)
        output_name = f"{record.source.name}__{clean_name}"
        name_key = (record.output_split, output_name.lower())
        if name_key in used_names:
            suffix = hashlib.sha1(str(record.image_path).encode("utf-8")).hexdigest()[:10]
            output_name = f"{Path(output_name).stem}__{suffix}{Path(output_name).suffix}"
            name_key = (record.output_split, output_name.lower())
        used_names.add(name_key)

        destination_image = output_dir / "images" / record.output_split / output_name
        destination_label = output_dir / "labels" / record.output_split / f"{Path(output_name).stem}.txt"
        copied, clahe_applied = _copy_image(
            record.image_path,
            destination_image,
            record.output_split,
            clahe_mode,
            dark_threshold,
            hardlink,
        )
        if not copied:
            build_stats["skipped_unreadable_image"] += 1
            build_failures.append(str(record.image_path))
            continue
        destination_label.write_text("\n".join(labels.lines) + ("\n" if labels.lines else ""), encoding="utf-8")
        build_stats[f"images_{record.output_split}"] += 1
        build_stats["clahe_images"] += int(clahe_applied)
        build_stats["instances"] += len(labels.lines)
        if index % 2500 == 0:
            print(f"[prepare] {index}/{len(records)} records")

    yaml_path = output_dir / "dms_dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(_portable_yaml(output_dir), sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit["build"] = dict(sorted(build_stats.items()))
    audit["build_failures"] = build_failures
    audit["unknown_image_samples"] = [str(path) for path in unknown_images[:25]]
    audit_path = output_dir / "audit_report.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_weak_manifest(output_dir, weak_records)

    if make_zip:
        archive_path = output_dir.parent / f"{output_dir.name}.zip"
        shutil.make_archive(str(archive_path.with_suffix("")), "zip", output_dir)
        audit["archive"] = str(archive_path)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"Dataset YAML: {yaml_path}")
    return audit


def prepare_dms_dataset() -> dict:
    """Backward-compatible entry point used by the download scripts."""
    return build_dataset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DATASET_DIR)
    parser.add_argument("--split-mode", choices=("grouped", "preserve"), default="grouped")
    parser.add_argument("--clahe", choices=("none", "dark-only", "all"), default="dark-only")
    parser.add_argument("--dark-threshold", type=float, default=85.0)
    parser.add_argument("--copy", action="store_true", help="Copy unchanged images instead of using hard links")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        print(json.dumps(audit_dataset(args.raw_dir, args.split_mode), ensure_ascii=False, indent=2))
        return
    build_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        clahe_mode=args.clahe,
        dark_threshold=args.dark_threshold,
        hardlink=not args.copy,
        overwrite=args.overwrite,
        make_zip=args.make_zip,
    )


if __name__ == "__main__":
    main()
