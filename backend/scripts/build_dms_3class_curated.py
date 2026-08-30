"""Build the smoking-free, three-class DMS training dataset.

The source four-class dataset is never modified. Images containing a smoking
annotation are excluded, while the training split keeps a deterministic,
group-diverse subset of empty-label negatives. Validation and test retain all
non-smoking negatives so their difficulty is not artificially reduced.

Canonical class order:
0 phone, 1 seatbelt, 2 no-seatbelt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "dms_yolo_4class_v2"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dms_yolo_3class_v3_curated"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "phone", 1: "seatbelt", 2: "no-seatbelt"}
SMOKING_CLASS_ID = 3


def _source_name(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "unknown"


def _group_key(stem: str) -> str:
    """Collapse Roboflow augmentation hashes and DMS frames into capture groups."""
    source = _source_name(stem)
    base = stem.split("__", 1)[-1].lower()
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", base)
    if source == "dms_safety":
        base = re.split(r"_mp4[-_]", base, maxsplit=1)[0]
    return f"{source}:{base}"


def _stable_score(value: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _read_label(path: Path) -> tuple[list[str], list[int]]:
    lines: list[str] = []
    classes: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}: {raw!r}")
        class_id = int(parts[0])
        if class_id not in (*CLASS_NAMES.keys(), SMOKING_CLASS_ID):
            raise ValueError(f"Unknown class {class_id} at {path}:{line_number}")
        lines.append(line)
        classes.append(class_id)
    return lines, classes


def select_group_diverse_negatives(paths: list[Path], limit: int, seed: int) -> set[Path]:
    """Select negatives round-robin across capture groups with stable ordering."""
    if limit < 0 or len(paths) <= limit:
        return set(paths)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[_group_key(path.stem)].append(path)
    for key in groups:
        groups[key].sort(key=lambda path: _stable_score(path.name, seed))
    ordered_keys = sorted(groups, key=lambda key: _stable_score(key, seed))

    selected: set[Path] = set()
    depth = 0
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            candidates = groups[key]
            if depth < len(candidates):
                selected.add(candidates[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def _safe_output_name(image_path: Path, output: Path, split: str) -> tuple[str, bool]:
    """Shorten exceptionally long source names before Windows hits MAX_PATH."""
    candidate = image_path.name
    destination = output / "images" / split / candidate
    if os.name != "nt" or len(str(destination)) < 240:
        return candidate, False
    source = _source_name(image_path.stem)
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:12]
    readable = image_path.stem.split("__", 1)[-1][:72].rstrip(" ._-")
    return f"{source}__{readable}__{digest}{image_path.suffix.lower()}", True


def _prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source dataset or a child of it")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output}. Pass --overwrite to rebuild it.")
        processed_root = (PROJECT_ROOT / "data" / "processed").resolve()
        if processed_root not in output.parents:
            raise ValueError(f"Refusing to remove output outside {processed_root}: {output}")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_curated_dataset(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    train_negative_limit: int = 1200,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not (source / "dms_dataset.yaml").exists():
        raise FileNotFoundError(f"Source dataset YAML not found: {source}")
    _prepare_output(source, output, overwrite)

    report: dict = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "policy": {
            "classes": CLASS_NAMES,
            "exclude_any_image_containing_class": {SMOKING_CLASS_ID: "smoking"},
            "train_negative_limit": train_negative_limit,
            "negative_selection": "deterministic group-diverse round-robin",
            "seed": seed,
        },
        "splits": {},
        "class_instances": Counter(),
        "source_images": Counter(),
        "transfer": Counter(),
    }

    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        images_by_stem = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        positives: list[tuple[Path, list[str], list[int]]] = []
        negatives: list[Path] = []
        smoking_images = 0
        smoking_only = 0
        smoking_mixed = 0

        for label_path in sorted(label_dir.glob("*.txt")):
            lines, classes = _read_label(label_path)
            if SMOKING_CLASS_ID in classes:
                smoking_images += 1
                if set(classes) == {SMOKING_CLASS_ID}:
                    smoking_only += 1
                else:
                    smoking_mixed += 1
                continue
            if lines:
                positives.append((label_path, lines, classes))
            else:
                negatives.append(label_path)

        kept_negatives = (
            select_group_diverse_negatives(negatives, train_negative_limit, seed)
            if split == "train"
            else set(negatives)
        )
        selected = positives + [(path, [], []) for path in sorted(kept_negatives)]

        for label_path, lines, classes in selected:
            image_path = images_by_stem.get(label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"Image missing for label: {label_path}")
            output_name, shortened = _safe_output_name(image_path, output, split)
            destination_image = output / "images" / split / output_name
            destination_label = output / "labels" / split / f"{Path(output_name).stem}.txt"
            transfer_mode = _link_or_copy(image_path, destination_image)
            destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            report["transfer"][transfer_mode] += 1
            report["transfer"]["shortened_names"] += int(shortened)
            report["source_images"][_source_name(label_path.stem)] += 1
            for class_id in classes:
                report["class_instances"][CLASS_NAMES[class_id]] += 1

        report["splits"][split] = {
            "source_images": len(images_by_stem),
            "output_images": len(selected),
            "positive_images": len(positives),
            "negative_images_available": len(negatives),
            "negative_images_kept": len(kept_negatives),
            "negative_images_dropped": len(negatives) - len(kept_negatives),
            "smoking_images_excluded": smoking_images,
            "smoking_only_excluded": smoking_only,
            "smoking_mixed_excluded": smoking_mixed,
        }

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (output / "dms_dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for key in ("class_instances", "source_images", "transfer"):
        report[key] = dict(sorted(report[key].items()))
    report["images_total"] = sum(item["output_images"] for item in report["splits"].values())
    report_path = output / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Dataset YAML: {output / 'dms_dataset.yaml'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-negative-limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_curated_dataset(
        source=args.source,
        output=args.output,
        train_negative_limit=args.train_negative_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
