from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from scripts.prepare_dms_dataset import (  # noqa: E402
    deterministic_split,
    group_key_for,
    harmonize_label,
    source_for_filename,
)
from scripts.pseudo_label_weak_dms_sources import auc_phone_entries  # noqa: E402
from scripts.train_yolo11_dms import configure_ultralytics_settings, validate_dataset_yaml  # noqa: E402


class SourceMappingTests(unittest.TestCase):
    def test_specific_roboflow_prefix_wins(self):
        source = source_for_filename("raw_roboflow_seatbelt_mobile_example.jpg")
        self.assertIsNotNone(source)
        self.assertEqual(source.name, "roboflow_seatbelt_mobile")

    def test_primary_four_class_mapping(self):
        source = source_for_filename("raw_roboflow_example.jpg")
        self.assertEqual(source.label_map, {0: 2, 1: 0, 2: 1, 3: 3})

    def test_dms_safety_mapping_drops_eye_classes(self):
        source = source_for_filename("dms_safety_frame.jpg")
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "sample.txt"
            label.write_text(
                "0 0.5 0.5 0.2 0.2\n"
                "2 0.5 0.5 0.2 0.2\n"
                "3 0.4 0.4 0.1 0.1\n"
                "4 0.3 0.3 0.1 0.1\n",
                encoding="utf-8",
            )
            result = harmonize_label(label, source)
        self.assertEqual([line.split()[0] for line in result.lines], ["3", "0", "1"])
        self.assertEqual(result.dropped["open-eye"], 1)

    def test_augmented_variants_share_group_and_split(self):
        source = source_for_filename("raw_roboflow_a.jpg")
        left = group_key_for("raw_roboflow_driver01.rf.aaaaaaaa.jpg", source)
        right = group_key_for("raw_roboflow_driver01.rf.bbbbbbbb.jpg", source)
        self.assertEqual(left, right)
        self.assertEqual(deterministic_split(left), deterministic_split(right))


class WeakSourceTests(unittest.TestCase):
    def test_auc_uses_v2_phone_classes_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "auc.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("v2_cam1_cam2_ split_by_driver/Camera 1/train/c1/a.jpg", b"x")
                archive.writestr("v2_cam1_cam2_ split_by_driver/Camera 1/train/c0/b.jpg", b"x")
                archive.writestr("v1_cam1_no_split/Text Right/c.jpg", b"x")
            with zipfile.ZipFile(archive_path) as archive:
                selected = auc_phone_entries(archive)
        self.assertEqual(selected, ["v2_cam1_cam2_ split_by_driver/Camera 1/train/c1/a.jpg"])


class TrainingConfigTests(unittest.TestCase):
    def test_ultralytics_settings_skip_missing_raytune_key(self):
        class FakeSettings(dict):
            pass

        settings = FakeSettings({"wandb": True, "sync": True})
        applied = configure_ultralytics_settings(settings)
        self.assertEqual(applied, {"wandb": False, "sync": False})
        self.assertEqual(settings, {"wandb": False, "sync": False})

    def test_dataset_yaml_requires_four_canonical_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "path": ".",
                        "train": "images/train",
                        "val": "images/val",
                        "test": "images/test",
                        "nc": 4,
                        "names": {0: "phone", 1: "seatbelt", 2: "no-seatbelt", 3: "smoking"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            data = validate_dataset_yaml(path)
        self.assertEqual(data["nc"], 4)

    def test_notebook_checkpoint_cells_compile_and_use_persistent_targets(self):
        notebook_path = BACKEND_DIR / "driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        for index, source in enumerate(code_cells):
            compile(source, f"notebook_code_cell_{index}", "exec")
        source = "\n".join(code_cells)
        self.assertIn(r'H:\My Drive\project3_runs', source)
        self.assertIn('RCLONE_REMOTE = os.getenv("DMS_RCLONE_REMOTE", "gdrive:")', source)
        self.assertIn('os.environ["RCLONE_DRIVE_ROOT_FOLDER_ID"] = RCLONE_DRIVE_ROOT_FOLDER_ID', source)
        self.assertNotIn('f"gdrive,root_folder_id={DRIVE_FOLDER_ID}:"', source)
        self.assertIn('attached_yamls = sorted(INPUT_ROOT.rglob("dms_dataset.yaml"))', source)
        self.assertIn('attached_training_scripts = sorted(INPUT_ROOT.rglob("train_yolo11_dms.py"))', source)
        self.assertIn('seatbelt_real_unlabelled_runtime.zip', source)
        self.assertIn('"sm_120" not in TORCH_CUDA_ARCHES', source)
        self.assertIn('TOTAL_BUDGET_HOURS = float(os.getenv("DMS_TOTAL_BUDGET_HOURS", "5.5"))', source)
        self.assertIn('BASE_TRAIN_HOURS = float(os.getenv("DMS_BASE_TRAIN_HOURS", "4.5"))', source)
        self.assertIn('FINE_TRAIN_HOURS = float(os.getenv("DMS_FINE_TRAIN_HOURS", "0.5"))', source)
        self.assertIn('BATCH = 32 if IS_RTX_5090 else 8', source)
        self.assertIn('time=BASE_TRAIN_HOURS', source)
        self.assertIn('time=FINE_TRAIN_HOURS', source)
        self.assertIn('elif RCLONE_FALLBACK_ENABLED:', source)
        self.assertIn("def rclone_mkdir(remote: str):", source)
        self.assertIn('model.add_callback("on_fit_epoch_end", on_fit_epoch_end)', source)
        self.assertIn("SAVE_PERIOD = 1", source)
        self.assertIn("WORKERS = 4 if IS_RTX_5090 else 2", source)
        self.assertIn("AUTO_RESUME_FROM_STORE = True", source)
        self.assertIn("def stage_is_complete", source)
        self.assertIn("FINE_RESUME_CKPT", source)
        self.assertIn("def download_named", source)
        self.assertIn("raytune=False", source)
        self.assertIn("if key in available", source)
        self.assertNotIn("AUTO_RESUME_FROM_DRIVE", source)


if __name__ == "__main__":
    unittest.main()
