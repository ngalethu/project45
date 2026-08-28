"""
===================================================================================
SCRIPT TỰ ĐỘNG TẢI & TỔNG HỢP CÁC BỘ DỮ LIỆU CHUYÊN BIỆT (DMS DATASETS DOWNLOADER)
===================================================================================
Quản lý & Tải về toàn bộ các bộ dữ liệu từ Roboflow & Kaggle tương ứng 5 Công trình:
1. Roboflow Dataset v9 (phone-detect-svavs): 28,284 ảnh (Phone + Seatbelt + Smoking)
2. Kaggle DMS Safety Dataset (habbas11/dms-driver-monitoring-system): 12,723 ảnh
3. Kaggle AUC Distracted Driver (tejakalepalle/auc-distracted-driver-dataset-v1)
4. State Farm Distracted Driver (state-farm-distracted-driver-detection): 22,424 ảnh
5. Roboflow Seatbelt & Mobile (aiactive20092009-gmail-com/seat_belt-and-mobile)
6. Kaggle Seatbelt Photos (alexandresintes/seatbelt-detection-dataset-real-car-photos)
"""
from __future__ import annotations

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

backend_dir = Path(__file__).parents[1]
sys.path.insert(0, str(backend_dir))

# Roboflow API configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = "ladailoc-yzh0x"
ROBOFLOW_PROJECT = "phone-detect-svavs"
ROBOFLOW_VERSION = 9

RAW_DATASET_DIR = backend_dir.parent / "data" / "raw_roboflow"
TARGET_IMAGES_DIR = backend_dir.parent / "data" / "images"
TARGET_LABELS_DIR = backend_dir.parent / "data" / "labels"

# Danh sách tất cả các Link / Datasets phục vụ huấn luyện DMS:
DMS_DATASET_SOURCES = {
    "roboflow_v9_primary": {
        "type": "roboflow",
        "url": "https://universe.roboflow.com/ladailoc-yzh0x/phone-detect-svavs/dataset/9",
        "workspace": "ladailoc-yzh0x",
        "project": "phone-detect-svavs",
        "version": 9,
        "description": "Bộ dữ liệu chính 28,284 ảnh (Phone, Seatbelt, No-Seatbelt, Smoking)"
    },
    "roboflow_seatbelt_mobile": {
        "type": "roboflow",
        "url": "https://universe.roboflow.com/aiactive20092009-gmail-com/seat_belt-and-mobile",
        "workspace": "aiactive20092009-gmail-com",
        "project": "seat_belt-and-mobile",
        "version": 1,
        "description": "Bộ dữ liệu kết hợp dây an toàn & điện thoại"
    },
    "kaggle_dms_safety": {
        "type": "kaggle",
        "dataset_slug": "habbas11/dms-driver-monitoring-system",
        "url": "https://www.kaggle.com/datasets/habbas11/dms-driver-monitoring-system",
        "description": "DMS Safety Dataset (12,723 ảnh - Eye, Cigarette, Phone, Seatbelt)"
    },
    "kaggle_auc_distracted": {
        "type": "kaggle",
        "dataset_slug": "tejakalepalle/auc-distracted-driver-dataset-v1",
        "url": "https://www.kaggle.com/datasets/tejakalepalle/auc-distracted-driver-dataset-v1",
        "description": "AUC Distracted Driver Dataset (Tư thế lái xe & Dùng điện thoại)"
    },
    "kaggle_seatbelt_real": {
        "type": "kaggle",
        "dataset_slug": "alexandresintes/seatbelt-detection-dataset-real-car-photos",
        "url": "https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos",
        "description": "Ảnh cabin thực tế driver vắt dây an toàn (Overhead Camera)"
    },
    "kaggle_state_farm": {
        "type": "kaggle_competition",
        "competition_name": "state-farm-distracted-driver-detection",
        "url": "https://www.kaggle.com/c/state-farm-distracted-driver-detection",
        "description": "State Farm Distracted Driver (22,424 ảnh 10 tư thế lái xe)"
    }
}


def extract_roboflow_zip_if_needed():
    """Giải nén file roboflow.zip an toàn trên Windows với tiền tố \\?\\ hỗ trợ đường dẫn dài (>260 ký tự)"""
    zip_path = RAW_DATASET_DIR / "roboflow.zip"
    if zip_path.exists():
        print(f"[INFO] Tìm thấy archive dữ liệu: {zip_path}")
        print("[INFO] Đang giải nén bộ dữ liệu Roboflow v9 (Phone + Seatbelt + Smoking)...")
        try:
            target_str = "\\\\?\\" + str(RAW_DATASET_DIR.resolve())
            with zipfile.ZipFile(str(zip_path)) as z:
                z.extractall(target_str)
            print("[SUCCESS] Đã giải nén an toàn toàn bộ 28,284 file ảnh & nhãn!")
        except Exception as e:
            print(f"[WARNING] Lỗi khi giải nén bằng long-path: {e}. Thử giải nén tiêu chuẩn...")
            with zipfile.ZipFile(str(zip_path)) as z:
                z.extractall(RAW_DATASET_DIR)


def download_roboflow_v9():
    print("==========================================================================")
    print("TẢI & ĐỒNG BỘ BỘ DỮ LIỆU DMS TỪ ROBOFLOW UNIVERSE (VERSION 9)")
    print("==========================================================================")
    print(f"Workspace: {ROBOFLOW_WORKSPACE} | Project: {ROBOFLOW_PROJECT} | Version: {ROBOFLOW_VERSION}")
    
    zip_path = RAW_DATASET_DIR / "roboflow.zip"
    if zip_path.exists() or (RAW_DATASET_DIR / "train" / "images").exists():
        print(f"[INFO] Dữ liệu Roboflow v9 đã có sẵn tại: {RAW_DATASET_DIR}")
        extract_roboflow_zip_if_needed()
        return

    if not ROBOFLOW_API_KEY:
        print("[SKIP] Thiếu ROBOFLOW_API_KEY. Đặt biến môi trường thay vì lưu API key trong source code.")
        return

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)

        dataset = version.download("yolov8", location=str(RAW_DATASET_DIR))
        print(f"[SUCCESS] Đã tải thành công dataset Roboflow v9 về: {dataset.location}")

    except Exception as e:
        print(f"[ERROR] Không thể tải Roboflow dataset từ API: {e}")


def print_registered_sources():
    print("\n==========================================================================")
    print("DANH SÁCH TẤT CẢ CÁC BỘ DỮ LIỆU DMS ĐÃ GỘP TRONG D:\\.idea\\project4\\data\\raw_roboflow")
    print("==========================================================================")
    for key, info in DMS_DATASET_SOURCES.items():
        print(f"📌 [{key}] {info['description']}")
        print(f"   -> URL: {info['url']}")


def main():
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    download_roboflow_v9()
    print_registered_sources()
    
    print("\n---> Tiến hành chuẩn hóa dữ liệu & chạy tiền xử lý CLAHE...")
    from scripts.prepare_dms_dataset import prepare_dms_dataset
    prepare_dms_dataset()


if __name__ == "__main__":
    main()

