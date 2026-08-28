"""
Script kiem tra toan bo backend ve xu ly video (Cloud & Edge Video Pipeline).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import json

# Dam bao import duoc app module
backend_dir = Path(__file__).parents[1]
sys.path.insert(0, str(backend_dir))

import cv2
from fastapi.testclient import TestClient
from app.cloud.main_cloud import app
from app.cloud.media_detector import process_uploaded_video
from app.edge.pipeline_yolo_pose import EdgePipeline


def test_cloud_video_detector():
    print("\n==========================================")
    print("TEST SUITE 1: Cloud Media Detector (process_uploaded_video)")
    print("==========================================")
    
    sample_dir = Path("..") / "data" / "sample_videos"
    if not sample_dir.exists():
        sample_dir = Path("data") / "sample_videos"
        
    test_videos = [
        "phone_driver.mp4",
        "no_seatbelt_driver.mp4",
        "smoking_driver.mp4",
        "normal_driving.mp4",
        "test.mp4",
    ]
    
    results = []
    
    for vname in test_videos:
        vpath = sample_dir / vname
        if not vpath.exists():
            print(f"[SKIP] Video file not found: {vpath}")
            continue
            
        print(f"\n---> Testing video: {vname}")
        t0 = time.time()
        try:
            res = process_uploaded_video(str(vpath), max_frames=60)
            elapsed = time.time() - t0
            
            print(f"  [SUCCESS] Elapsed: {elapsed:.2f}s")
            print(f"  - Event detected: {res.get('event_type')}")
            print(f"  - Max Confidence: {res.get('confidence')}")
            print(f"  - Processed frames: {res.get('total_frames')}")
            print(f"  - Frame path: {res.get('frame_path')}")
            print(f"  - Clip path: {res.get('clip_path')}")
            
            # Check output files existence
            f_exists = Path(res.get('frame_path', '')).exists() if res.get('frame_path') else False
            c_exists = Path(res.get('clip_path', '')).exists() if res.get('clip_path') else False
            
            print(f"  - Output Frame exists: {f_exists}")
            print(f"  - Output Clip exists: {c_exists}")
            
            results.append({
                "video": vname,
                "status": "PASSED" if (f_exists and c_exists) else "WARNING",
                "event_type": res.get("event_type"),
                "confidence": res.get("confidence"),
                "frames": res.get("total_frames"),
                "time_sec": round(elapsed, 2)
            })
        except Exception as e:
            print(f"  [FAILED] Error: {e}")
            results.append({
                "video": vname,
                "status": f"FAILED: {e}",
            })
            
    return results


def test_cloud_fastapi_upload():
    print("\n==========================================")
    print("TEST SUITE 2: Cloud FastAPI Endpoint (/api/auto_detect_media)")
    print("==========================================")
    
    client = TestClient(app)
    
    # Check Health
    health_resp = client.get("/health")
    print(f"GET /health: {health_resp.status_code} - {health_resp.json()}")
    assert health_resp.status_code == 200
    
    sample_dir = Path("..") / "data" / "sample_videos"
    if not sample_dir.exists():
        sample_dir = Path("data") / "sample_videos"
        
    test_video = sample_dir / "test.mp4"
    if not test_video.exists():
        print(f"[SKIP] {test_video} not found")
        return False
        
    print(f"---> Uploading video {test_video.name} to /api/auto_detect_media")
    with open(test_video, "rb") as f:
        response = client.post(
            "/api/auto_detect_media",
            files={"media_file": ("test.mp4", f, "video/mp4")},
            data={"notes": "Automated backend test video upload"}
        )
        
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("  [SUCCESS] Alert created successfully:")
        print(f"  - Alert ID: {data.get('alert', {}).get('id')}")
        print(f"  - Event Type: {data.get('alert', {}).get('event_type')}")
        print(f"  - Source Device: {data.get('alert', {}).get('source_device')}")
        print(f"  - Frame URL: {data.get('alert', {}).get('frame_url')}")
        print(f"  - Clip URL: {data.get('alert', {}).get('clip_url')}")
        return True
    else:
        print(f"  [FAILED] Detail: {response.text}")
        return False


def test_edge_pipeline_video():
    print("\n==========================================")
    print("TEST SUITE 3: Edge Video Pipeline (EdgePipeline)")
    print("==========================================")
    
    sample_dir = Path("..") / "data" / "sample_videos"
    if not sample_dir.exists():
        sample_dir = Path("data") / "sample_videos"
        
    test_video = sample_dir / "phone_driver.mp4"
    if not test_video.exists():
        test_video = sample_dir / "test.mp4"
        
    print(f"---> Running EdgePipeline on source: {test_video}")
    
    pipeline = EdgePipeline(config_path="config.yaml")
    
    # Dynamic override to prevent GUI popup blocking
    pipeline.cfg["edge"]["show_window"] = False
    pipeline.cfg["edge"]["save_video"] = True
    
    t0 = time.time()
    try:
        pipeline.run(source_override=str(test_video), send_to_cloud_override=False)
        elapsed = time.time() - t0
        print(f"  [SUCCESS] Edge pipeline executed in {elapsed:.2f}s")
        
        output_video = Path(pipeline.cfg["edge"]["output_video_path"])
        if output_video.exists():
            size_kb = output_video.stat().st_size / 1024
            print(f"  - Rendered output video generated: {output_video} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  - Output video file missing: {output_video}")
            return False
    except Exception as e:
        print(f"  [FAILED] Edge pipeline error: {e}")
        return False


if __name__ == "__main__":
    print("Starting Backend Video Processing Tests...")
    t_start = time.time()
    
    res1 = test_cloud_video_detector()
    res2 = test_cloud_fastapi_upload()
    res3 = test_edge_pipeline_video()
    
    total_time = time.time() - t_start
    print("\n==========================================")
    print("SUMMARY OF BACKEND VIDEO TESTS")
    print("==========================================")
    print(f"Total Test Time: {total_time:.2f} seconds\n")
    print("Test Suite 1 (Cloud Detector):")
    for r in res1:
        print(f"  - {r['video']}: {r['status']} | Event: {r.get('event_type')} | Conf: {r.get('confidence')} | Frames: {r.get('frames')}")
        
    print(f"\nTest Suite 2 (FastAPI Video Upload): {'PASSED' if res2 else 'FAILED'}")
    print(f"Test Suite 3 (Edge Video Pipeline): {'PASSED' if res3 else 'FAILED'}")
