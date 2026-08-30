"""Run the event-level DMS video benchmark on a Kaggle GPU."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


BENCHMARK_PIPELINE_VERSION = "real20-adaptive-v1"

INPUT_ROOT = Path("/kaggle/input")
PROJECT_ROOT = Path("/kaggle/working/project")
EXPORT_ROOT = Path("/kaggle/working/dms_video_benchmark")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


import torch


def ensure_cuda_compatibility() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    print("CUDA compatibility:", torch.cuda.get_device_name(0), required_arch, sorted(supported_arches))
    if required_arch in supported_arches:
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--force-reinstall",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_cuda_compatibility()
if os.environ.get("DMS_RUNTIME_DEPS_READY") != "1":
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ultralytics",
            "mediapipe==0.10.21",
            "opencv-python-headless",
            "pyyaml",
        ],
        check=True,
    )
    # Legacy MediaPipe Solutions pins Protobuf 4.x, while Kaggle's preinstalled
    # TensorFlow 2.20 imports Protobuf 5.x symbols. TensorFlow is optional here.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-cpu"],
        check=False,
    )
    # pip replaces NumPy/Protobuf files used by the running Kaggle process.
    # Restart once so MediaPipe imports from a coherent, freshly loaded environment.
    os.environ["DMS_RUNTIME_DEPS_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

code_zips = sorted(INPUT_ROOT.rglob("benchmark_code.zip"))
if code_zips:
    with zipfile.ZipFile(code_zips[0]) as archive:
        archive.extractall(PROJECT_ROOT)
else:
    # Kaggle expands uploaded ZIP files into a directory during dataset creation.
    extracted_markers = sorted(
        INPUT_ROOT.rglob("benchmark_code/backend/scripts/evaluate_sample_videos.py")
    )
    assert extracted_markers, "Neither benchmark_code.zip nor its extracted tree was mounted"
    extracted_backend = extracted_markers[0].parents[1]
    shutil.copytree(extracted_backend, PROJECT_ROOT / "backend", dirs_exist_ok=True)

sample_root = PROJECT_ROOT / "data" / "sample_videos"
sample_root.mkdir(parents=True, exist_ok=True)
manifest_candidates = sorted(INPUT_ROOT.rglob("benchmark_manifest.json"))
assert manifest_candidates, "benchmark_manifest.json was not mounted"
manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
shutil.copy2(manifest_candidates[0], sample_root / "benchmark_manifest.json")
for sample in manifest.get("videos") or []:
    candidates = sorted(INPUT_ROOT.rglob(sample["file"]))
    assert candidates, f"Missing benchmark video: {sample['file']}"
    shutil.copy2(candidates[0], sample_root / sample["file"])

best_candidates = sorted(INPUT_ROOT.rglob("best.pt"))
assert best_candidates, "best.pt checkpoint was not mounted"
model_dir = PROJECT_ROOT / "backend" / "models"
model_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(best_candidates[0], model_dir / "best.pt")

vehicle_candidates = sorted(INPUT_ROOT.rglob("yolo11m.pt"))
assert vehicle_candidates, "yolo11m.pt vehicle checkpoint was not mounted"
shutil.copy2(vehicle_candidates[0], PROJECT_ROOT / "backend" / "yolo11m.pt")

backend_dir = PROJECT_ROOT / "backend"
os.chdir(backend_dir)
sys.path.insert(0, str(backend_dir))
from scripts.evaluate_sample_videos import OUTPUT_PATH, evaluate_manifest


payload = evaluate_manifest(sample_root / "benchmark_manifest.json")
result_path = EXPORT_ROOT / "video_evaluation_metrics.json"
result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
provenance_path = best_candidates[0].parent / "provenance.json"
if provenance_path.is_file():
    shutil.copy2(provenance_path, EXPORT_ROOT / "model_provenance.json")
print("BENCHMARK_OUTPUT", result_path)
print("PIPELINE_OUTPUT", OUTPUT_PATH)
print("BENCHMARK_PIPELINE_VERSION", BENCHMARK_PIPELINE_VERSION)
