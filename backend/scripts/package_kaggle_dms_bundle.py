"""Create a Kaggle-uploadable bundle for the DMS training notebook."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "dms_yolo_3class_v4_12k"
DEFAULT_MODEL = BACKEND_DIR / "yolo11m.pt"
DEFAULT_NOTEBOOK = BACKEND_DIR / "kaggle_train_dms_3class_12k.ipynb"
DEFAULT_OUTPUT = BACKEND_DIR / "outputs" / "kaggle_dms_3class_v4_12k_bundle"


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def zip_dataset(dataset_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    files = sorted(path for path in dataset_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            relative = Path(dataset_dir.name) / path.relative_to(dataset_dir)
            compression = zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else zipfile.ZIP_DEFLATED
            archive.write(path, relative.as_posix(), compress_type=compression, compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None)
            if index % 5000 == 0:
                print(f"[zip] {index}/{len(files)} files")


def zip_training_code(archive_path: Path) -> None:
    scripts = (
        BACKEND_DIR / "scripts" / "train_yolo11_dms.py",
        BACKEND_DIR / "scripts" / "install_kaggle_dms_model.py",
        BACKEND_DIR / "scripts" / "build_dms_3class_12k.py",
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for script in scripts:
            archive.write(script, script.name)


def build_bundle(
    dataset_dir: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    notebook_path: Path = DEFAULT_NOTEBOOK,
    output_dir: Path = DEFAULT_OUTPUT,
    kaggle_username: str = "YOUR_KAGGLE_USERNAME",
) -> dict:
    required = (dataset_dir, model_path, notebook_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Older bundle builds included the restricted AUC archive. Remove only that
    # exact generated bundle entry; the source archive under data/sources stays.
    restricted_bundle_copy = output_dir / "auc.distracted.driver.dataset_v2.zip"
    if restricted_bundle_copy.exists():
        restricted_bundle_copy.unlink()

    dataset_zip = output_dir / f"{dataset_dir.name}.zip"
    zip_dataset(dataset_dir, dataset_zip)
    _link_or_copy(model_path, output_dir / "yolo11m.pt")
    shutil.copy2(notebook_path, output_dir / notebook_path.name)
    zip_training_code(output_dir / "training_code.zip")
    shutil.copy2(dataset_dir / "audit_report.json", output_dir / "audit_report.json")
    for manifest_path in (
        PROJECT_ROOT / "data" / "sources" / "multidomain_source_manifest.json",
        PROJECT_ROOT / "data" / "evaluation" / "seatbelt_real_external_manifest.json",
    ):
        if manifest_path.exists():
            shutil.copy2(manifest_path, output_dir / manifest_path.name)

    metadata = {
        "title": "DMS YOLO11 Three Class 12K",
        "id": f"{kaggle_username}/dms-yolo11-three-class-12k",
        "licenses": [{"name": "other"}],
        "subtitle": "Leakage-aware 12k phone seatbelt and no-seatbelt bundle",
        "description": (
            "Smoking-free three-class DMS bundle with exactly 12,000 training images, "
            "one image per capture group, group-disjoint splits and curated negatives. "
            "Source licenses remain applicable; see the notebook and audit report."
        ),
        "keywords": ["computer-vision", "object-detection", "driver-monitoring", "yolo"],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            files.append({"name": path.name, "bytes": path.stat().st_size})
    manifest = {
        "bundle_dir": str(output_dir.resolve()),
        "kaggle_id_requires_edit": kaggle_username == "YOUR_KAGGLE_USERNAME",
        "files": files,
        "upload_command": f'kaggle datasets create -p "{output_dir.resolve()}" -r skip',
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kaggle-username", default="YOUR_KAGGLE_USERNAME")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_bundle(
        dataset_dir=args.dataset_dir,
        model_path=args.model,
        notebook_path=args.notebook,
        output_dir=args.output_dir,
        kaggle_username=args.kaggle_username,
    )
