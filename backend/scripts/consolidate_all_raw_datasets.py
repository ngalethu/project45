"""
===================================================================================
SCRIPT GỘP VẬT LÝ TẤT CẢ 6 BỘ DỮ LIỆU DMS VÀO 1 THƯ MỤC DUY NHẤT (RAW_ALL_DMS)
===================================================================================
Tải về & hợp nhất toàn bộ dữ liệu từ 6 nguồn:
1. Roboflow Dataset v9 (28,284 ảnh)
2. Roboflow Seatbelt & Mobile (779+ ảnh)
3. Kaggle DMS Safety Dataset (habbas11/dms-driver-monitoring-system - 12,723 ảnh)
4. Kaggle AUC Distracted Driver (tejakalepalle/auc-distracted-driver-dataset-v1)
5. Kaggle Seatbelt Real Car Photos (alexandresintes/seatbelt-detection-dataset-real-car-photos)
6. Kaggle State Farm Distracted Driver (state-farm-distracted-driver-detection)

Tất cả ảnh & nhãn được copy trực tiếp vào 1 THƯ MỤC DUY NHẤT:
-> D:\.idea\project4\data\raw_all_dms\
"""
from __future__ import annotations

import os
import sys
import shutil
import zipfile
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Force UTF-8 stdout encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

backend_dir = Path(__file__).parents[1]
sys.path.insert(0, str(backend_dir))

DATASET_ROOT = backend_dir.parent / "data"
RAW_MASTER_DIR = DATASET_ROOT / "raw"
OUTPUT_PREPARED_DIR = DATASET_ROOT / "processed" / "dms_prepared_dataset"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


def download_additional_sources():
    print("==========================================================================")
    print("1. TẢI VỀ & ĐỒNG BỘ CẢ 6 NGUỜI DỮ LIỆU THÔ VÀO 1 THƯ MỤC 'data/raw'")
    print("==========================================================================")

    # 1. Roboflow Seatbelt & Mobile
    sb_mob_dir = DATASET_ROOT / "raw_temp_seatbelt_mobile"
    if not sb_mob_dir.exists():
        if not ROBOFLOW_API_KEY:
            print("[SKIP] Thiếu ROBOFLOW_API_KEY; không tải Roboflow Seatbelt & Mobile.")
        else:
            try:
                print("[DOWNLOAD] Đang tải Roboflow Seatbelt & Mobile...")
                from roboflow import Roboflow
                rf = Roboflow(api_key=ROBOFLOW_API_KEY)
                project = rf.workspace("aiactive20092009-gmail-com").project("seat_belt-and-mobile")
                project.version(1).download("yolov8", location=str(sb_mob_dir))
            except Exception as e:
                print(f"[WARNING] Lỗi tải Roboflow seatbelt mobile: {e}")

    # 2. Kaggle Datasets
    try:
        import kagglehub
        kaggle_datasets = [
            ("dms_safety", "habbas11/dms-driver-monitoring-system"),
            ("auc_distracted", "tejakalepalle/auc-distracted-driver-dataset-v1"),
            ("seatbelt_real", "alexandresintes/seatbelt-detection-dataset-real-car-photos")
        ]
        for name, slug in kaggle_datasets:
            try:
                print(f"[DOWNLOAD] Đang đồng bộ Kaggle dataset: {slug}...")
                kagglehub.dataset_download(slug)
            except Exception as ke:
                print(f"[NOTE] Thông báo Kaggle {slug}: {ke}")
    except Exception as e:
        print(f"[NOTE] Kagglehub notice: {e}")


def consolidate_into_single_raw_folder(cleanup_sources: bool = False):
    print("\n==========================================================================")
    print("2. HỢP NHẤT TẤT CẢ VÀO DUY NHẤT 1 THƯ MỤC RAW: 'data/raw'")
    print("==========================================================================")

    for split in ["train", "val", "test"]:
        (RAW_MASTER_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (RAW_MASTER_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    sources = [
        DATASET_ROOT / "raw_roboflow",
        DATASET_ROOT / "raw_roboflow_seatbelt_mobile",
        DATASET_ROOT / "raw_all_dms",
        DATASET_ROOT / "raw_temp_seatbelt_mobile"
    ]

    kaggle_cache = Path(os.path.expanduser("~")) / ".cache" / "kagglehub" / "datasets"
    if kaggle_cache.exists():
        for k_dir in kaggle_cache.glob("**/*"):
            if k_dir.is_dir() and any(k_dir.glob("*.jpg")):
                sources.append(k_dir)

    total_merged_files = 0
    for src in sources:
        if not src.exists():
            continue

        print(f"[MERGE] Đang gộp dữ liệu từ: {src.name} -> data/raw...")
        for s_in, s_out in [("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")]:
            img_dir = src / s_in / "images" if (src / s_in / "images").exists() else (src / s_in if (src / s_in).exists() else src)
            lbl_dir = src / s_in / "labels" if (src / s_in / "labels").exists() else (src / s_in if (src / s_in).exists() else src)

            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
            for img_path in img_files:
                dest_img = RAW_MASTER_DIR / s_out / "images" / f"{src.name}_{img_path.name}"
                dest_lbl = RAW_MASTER_DIR / s_out / "labels" / f"{src.name}_{img_path.stem}.txt"

                if not dest_img.exists():
                    try:
                        shutil.copy(str(img_path), str(dest_img))
                        total_merged_files += 1
                    except Exception:
                        pass

                src_lbl = lbl_dir / f"{img_path.stem}.txt"
                if src_lbl.exists() and not dest_lbl.exists():
                    try:
                        shutil.copy(str(src_lbl), str(dest_lbl))
                    except Exception:
                        pass

    # Giữ nguyên source mặc định để có thể audit/rebuild. Chỉ xóa khi caller
    # truyền cleanup_sources=True một cách tường minh.
    if cleanup_sources:
        for src in [DATASET_ROOT / "raw_roboflow", DATASET_ROOT / "raw_roboflow_seatbelt_mobile", DATASET_ROOT / "raw_all_dms", DATASET_ROOT / "raw_temp_seatbelt_mobile"]:
            if src.exists() and src != RAW_MASTER_DIR:
                try:
                    shutil.rmtree(src)
                    print(f"[CLEANUP] Đã xóa thư mục phụ: {src.name}")
                except Exception as e:
                    print(f"[WARNING] Không thể xóa {src.name}: {e}")

    print(f"\n[HOÀN THÀNH GỘP DUY NHẤT 1 THƯ MỤC RAW]")
    print(f"Thư mục raw duy nhất: {RAW_MASTER_DIR}")
    print(f" - Train: {len(list((RAW_MASTER_DIR / 'train' / 'images').glob('*')))} ảnh")
    print(f" - Val  : {len(list((RAW_MASTER_DIR / 'val' / 'images').glob('*')))} ảnh")
    print(f" - Test : {len(list((RAW_MASTER_DIR / 'test' / 'images').glob('*')))} ảnh")



def main():
    download_additional_sources()
    consolidate_into_single_raw_folder()

    print("\n---> Chạy chuẩn hóa dữ liệu & tạo Zip master từ duy nhất 1 thư mục 'data/raw'...")
    from scripts.prepare_dms_dataset import prepare_dms_dataset
    prepare_dms_dataset()


if __name__ == "__main__":
    main()
