"""
Script gia lap Frontend upload Anh va Video (co ten tieng Viet, khoang trang) de test backend endpoint /api/auto_detect_media.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 stdout encoding for Windows PowerShell console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

backend_dir = Path(__file__).parents[1]
sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.cloud.main_cloud import app


def create_dummy_image_bytes():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (34, 197, 94)
    cv2.putText(img, "Test Upload Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def run_frontend_upload_test():
    print("==========================================")
    print("MO PHONG FRONTEND UPLOAD FILE ANH & VIDEO LEN BACKEND")
    print("==========================================")

    client = TestClient(app)

    # 1. Test Upload Anh voi ten tieng Viet co dau & khoang trang
    print("\n1. Test Upload Anh (Ten Tieng Viet co dau & khoang trang)")
    img_bytes = create_dummy_image_bytes()
    
    resp_img = client.post(
        "/api/auto_detect_media",
        files={"media_file": ("anh_lai_xe_dung_dien_thoai_thu_nghiem.jpg", img_bytes, "image/jpeg")},
        data={"notes": "Upload tu Web UI Frontend (Test tieng Viet)"}
    )
    
    print(f"Status Code: {resp_img.status_code}")
    if resp_img.status_code == 200:
        data = resp_img.json()
        print("  [SUCCESS] Ket qua tra ve cho Frontend:")
        print(f"  - Alert ID: #{data['alert']['id']}")
        print(f"  - Event Type: {data['detection']['event_type']}")
        print(f"  - Confidence: {data['detection']['confidence']}")
        print(f"  - Frame URL: {data['detection']['frame_url']}")
    else:
        print(f"  [FAILED] Response: {resp_img.text}")

    # 2. Test Upload Video voi ten file tieng Viet & khoang trang
    print("\n2. Test Upload Video (Ten file Tieng Viet & khoang trang)")
    sample_video = Path("..") / "data" / "sample_videos" / "test.mp4"
    if not sample_video.exists():
        sample_video = Path("data") / "sample_videos" / "test.mp4"

    if sample_video.exists():
        with open(sample_video, "rb") as f:
            v_bytes = f.read()

        resp_vid = client.post(
            "/api/auto_detect_media",
            files={"media_file": ("video lai xe thuc te 1080p.mp4", v_bytes, "video/mp4")},
            data={"notes": "Upload Video tu Web UI Frontend"}
        )

        print(f"Status Code: {resp_vid.status_code}")
        if resp_vid.status_code == 200:
            data = resp_vid.json()
            print("  [SUCCESS] Ket qua tra ve cho Frontend:")
            print(f"  - Alert ID: #{data['alert']['id']}")
            print(f"  - Event Type: {data['detection']['event_type']}")
            print(f"  - Total Frames: {data['detection']['total_frames']}")
            print(f"  - Frame URL: {data['detection']['frame_url']}")
            print(f"  - Clip URL: {data['detection']['clip_url']}")
        else:
            print(f"  [FAILED] Response: {resp_vid.text}")
    else:
        print(f"[SKIP] Video {sample_video} ko tim thay")

    # 3. Test Truy Van Danh Sach Alerts tu API Frontend (/api/alerts)
    print("\n3. Test Frontend Lay Danh Sach Alerts (/api/alerts)")
    resp_list = client.get("/api/alerts?limit=5")
    print(f"Status Code: {resp_list.status_code}")
    if resp_list.status_code == 200:
        alerts = resp_list.json()
        print(f"  [SUCCESS] Tong so alerts trong DB: {alerts['total']}")
        if alerts['items']:
            top = alerts['items'][0]
            print(f"  - Top 1 alert gan nhat: ID #{top['id']} | Event: {top['event_type']} | Clip URL: {top.get('clip_url')}")
    else:
        print(f"  [FAILED] Response: {resp_list.text}")

    print("\n==========================================")
    print("TEST HOAN TAT TROI CHAY!")
    print("==========================================")


if __name__ == "__main__":
    run_frontend_upload_test()
