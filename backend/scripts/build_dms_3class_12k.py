"""Build a leakage-aware 12k-train, three-class DMS dataset.

The source is the smoking-free v3 dataset.  Validation and test are copied
unchanged.  Training keeps at most one image per capture/augmentation group,
retains all rare no-seatbelt and roadside/windshield groups, caps negatives,
and fills the remaining budget with balanced phone/seatbelt strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "dms_yolo_3class_v3_curated"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dms_yolo_3class_v4_12k"
CLASS_NAMES = {0: "phone", 1: "seatbelt", 2: "no-seatbelt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
ROADSIDE_SOURCE = "roboflow_seatbelt_mobile"


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    source: str
    group: str
    classes: tuple[int, ...]
    instances: tuple[int, ...]


def _source_name(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "unknown"


def _group_key(stem: str) -> str:
    source = _source_name(stem)
    base = stem.split("__", 1)[-1].lower()
    base = re.sub(r"\.rf\.[0-9a-f]+$", "", base)
    if source == "dms_safety":
        base = re.split(r"_mp4[-_]", base, maxsplit=1)[0]
    return f"{source}:{base}"


def _stable(value: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _read_classes(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    instances: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}: {raw!r}")
        class_id = int(parts[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"Unexpected class {class_id} at {path}:{line_number}")
        instances.append(class_id)
    return tuple(sorted(set(instances))), tuple(instances)


def _records(source_root: Path, split: str) -> list[Record]:
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    records: list[Record] = []
    for label in sorted(label_dir.glob("*.txt")):
        image = images.get(label.stem)
        if image is None:
            raise FileNotFoundError(f"Image missing for {label}")
        classes, instances = _read_classes(label)
        records.append(
            Record(
                image=image,
                label=label,
                source=_source_name(label.stem),
                group=_group_key(label.stem),
                classes=classes,
                instances=instances,
            )
        )
    if len(records) != len(images):
        raise ValueError(f"Unpaired files in {split}: images={len(images)} labels={len(records)}")
    return records


def _representatives(records: list[Record], seed: int) -> list[Record]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.group].append(record)
    chosen: list[Record] = []
    for group, candidates in groups.items():
        # Stable representative avoids selecting several Roboflow augmentations
        # of the same capture while keeping the build reproducible.
        candidates.sort(key=lambda item: _stable(item.image.name, seed))
        chosen.append(candidates[0])
    return sorted(chosen, key=lambda item: _stable(item.group, seed))


def select_train(records: list[Record], target: int, negative_limit: int, seed: int) -> list[Record]:
    reps = _representatives(records, seed)
    if len(reps) < target:
        raise ValueError(f"Only {len(reps)} unique groups are available for target={target}")

    negatives = [record for record in reps if not record.classes]
    negatives.sort(key=lambda item: _stable(item.group, seed + 11))
    allowed_negative_groups = {record.group for record in negatives[:negative_limit]}

    required: dict[str, Record] = {}
    for record in reps:
        if (
            record.source in {ROADSIDE_SOURCE, "dms_safety"}
            or 2 in record.classes
            or len(record.classes) > 1
            or record.group in allowed_negative_groups
        ):
            required[record.group] = record
    if len(required) > target:
        raise ValueError(f"Required coverage ({len(required)}) exceeds target ({target})")

    # Fill remaining slots round-robin across source and label-combination strata.
    buckets: dict[tuple[str, tuple[int, ...]], deque[Record]] = defaultdict(deque)
    for record in reps:
        if record.group in required or not record.classes:
            continue
        buckets[(record.source, record.classes)].append(record)
    keys = sorted(buckets, key=lambda key: _stable(repr(key), seed + 29))
    selected = dict(required)
    while len(selected) < target:
        added = False
        for key in keys:
            if not buckets[key]:
                continue
            record = buckets[key].popleft()
            selected[record.group] = record
            added = True
            if len(selected) == target:
                break
        if not added:
            raise RuntimeError(f"Unable to fill train target; selected={len(selected)}")
    return sorted(selected.values(), key=lambda item: _stable(item.group, seed))


def _prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    processed_root = (PROJECT_ROOT / "data" / "processed").resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source or its child")
    if processed_root not in output.parents:
        raise ValueError(f"Output must be under {processed_root}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {output}; pass --overwrite")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def _transfer(records: list[Record], output: Path, split: str, stats: Counter) -> None:
    for record in records:
        image_out = output / "images" / split / record.image.name
        label_out = output / "labels" / split / record.label.name
        stats[_link_or_copy(record.image, image_out)] += 1
        shutil.copy2(record.label, label_out)


def _summarise(records: list[Record]) -> dict:
    instances = Counter(class_id for record in records for class_id in record.instances)
    combos = Counter("negative" if not record.classes else "+".join(CLASS_NAMES[c] for c in record.classes) for record in records)
    return {
        "images": len(records),
        "unique_groups": len({record.group for record in records}),
        "sources": dict(sorted(Counter(record.source for record in records).items())),
        "class_instances": {CLASS_NAMES[c]: instances[c] for c in CLASS_NAMES},
        "label_combinations": dict(sorted(combos.items())),
        "negative_images": sum(not record.classes for record in records),
    }


def build_dataset(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    train_target: int = 12_000,
    negative_limit: int = 800,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not (source / "dms_dataset.yaml").exists():
        raise FileNotFoundError(f"Dataset YAML missing: {source}")
    _prepare_output(source, output, overwrite)

    source_records = {split: _records(source, split) for split in SPLITS}
    selected = {
        "train": select_train(source_records["train"], train_target, negative_limit, seed),
        "val": source_records["val"],
        "test": source_records["test"],
    }
    transfer = Counter()
    for split in SPLITS:
        _transfer(selected[split], output, split, transfer)

    yaml_payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (output / "dms_dataset.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in selected.items():
        for record in records:
            group_splits[record.group].add(split)
    crossing = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    roadside_test = [record for record in selected["test"] if record.source == ROADSIDE_SOURCE]
    external_manifest = {
        "purpose": "Independent windshield/roadside-domain evaluation; never used for training",
        "source": ROADSIDE_SOURCE,
        "license": "CC BY 4.0",
        "source_url": "https://universe.roboflow.com/aiactive20092009-gmail-com/seat_belt-and-mobile",
        "images": [
            {
                "image": f"images/test/{record.image.name}",
                "label": f"labels/test/{record.label.name}",
                "camera_id": record.group,
                "driver_id": "unknown",
            }
            for record in roadside_test
        ],
    }
    (output / "external_test_manifest.json").write_text(
        json.dumps(external_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_assurance = {
        "capture_group_disjoint": len(crossing) == 0,
        "groups_crossing_splits": len(crossing),
        "camera_disjoint_proxy": "capture/source group key",
        "driver_disjoint": "not_verifiable",
        "reason": "Most public sources do not provide driver identity metadata.",
        "required_future_collection_fields": ["camera_id", "driver_id", "capture_session_id"],
    }
    (output / "split_assurance.json").write_text(
        json.dumps(split_assurance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "source_dataset": str(source),
        "output_dataset": str(output),
        "policy": {
            "classes": CLASS_NAMES,
            "train_target": train_target,
            "one_image_per_capture_group": True,
            "retain_all_sources": [ROADSIDE_SOURCE, "dms_safety"],
            "retain_all_images_containing": ["no-seatbelt"],
            "train_negative_limit": negative_limit,
            "selection": "required coverage plus deterministic round-robin strata",
            "seed": seed,
        },
        "splits": {split: _summarise(records) for split, records in selected.items()},
        "group_leakage": {
            "groups_crossing_splits": len(crossing),
            "samples": crossing[:20],
        },
        "driver_leakage": "not_verifiable because source driver IDs are unavailable",
        "roadside_external_test_images": len(roadside_test),
        "transfer": dict(sorted(transfer.items())),
        "images_total": sum(len(records) for records in selected.values()),
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-target", type=int, default=12_000)
    parser.add_argument("--negative-limit", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        source=args.source,
        output=args.output,
        train_target=args.train_target,
        negative_limit=args.negative_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )
