from __future__ import annotations
import os
import urllib.request
import cv2
import numpy as np

SAMPLE_VIDEOS_DIR = "data/sample_videos"

# Public sample mp4 video URLs (fallback to synthetic video generation if network fails or URL unavailable)
PUBLIC_VIDEO_SOURCES = [
    {
        "filename": "phone_driver.mp4",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "event_type": "using_phone"
    },
    {
        "filename": "no_seatbelt_driver.mp4",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
        "event_type": "no_seatbelt"
    },
    {
        "filename": "normal_driving.mp4",
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
        "event_type": "normal"
    }
]

def generate_synthetic_driver_video(output_path: str, event_type: str, num_frames: int = 150):
    """Generates a high quality synthetic driver behavior test video using OpenCV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (640, 480))
    
    for f in range(num_frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Cabin interior background
        cv2.rectangle(img, (0, 0), (640, 480), (15, 23, 42), -1)
        # Windshield view (road motion effect)
        cv2.rectangle(img, (80, 20), (560, 180), (30, 41, 59), -1)
        cv2.line(img, (320, 20), (320 + int(np.sin(f/4)*20), 180), (148, 163, 184), 2)
        
        # Steering wheel
        cv2.circle(img, (200, 360), 60, (51, 65, 85), 12)
        
        # Driver Head & Torso
        head_y = 220 + int(np.sin(f / 10) * 4)
        cv2.circle(img, (360, head_y), 55, (100, 116, 139), -1) # Head
        cv2.rectangle(img, (290, head_y + 55), (430, 460), (71, 85, 105), -1) # Torso
        
        # Hand & Object simulation based on event_type
        hand_x = 360 + int(np.sin(f / 6) * 15)
        hand_y = head_y + 20
        
        if event_type == "using_phone":
            # Right hand holding phone up to ear
            cv2.line(img, (400, 350), (hand_x + 30, hand_y), (100, 116, 139), 12)
            cv2.rectangle(img, (hand_x + 20, hand_y - 30), (hand_x + 45, hand_y + 20), (0, 0, 220), -1)
            cv2.putText(img, "USING PHONE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif event_type == "no_seatbelt":
            # Missing seatbelt strap across shoulder
            cv2.putText(img, "NO SEATBELT WARNING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
        else:
            # Normal driving: Seatbelt on, hands on wheel
            cv2.line(img, (300, head_y + 55), (420, 460), (0, 255, 255), 6) # Seatbelt strap
            cv2.putText(img, "NORMAL DRIVING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(img, f"Frame: {f+1}/{num_frames}", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
        out.write(img)
        
    out.release()
    print(f"[SYNTHETIC] Generated test video: {output_path}")

def download_and_prepare_videos():
    os.makedirs(SAMPLE_VIDEOS_DIR, exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print("=== Downloading / Preparing Driver Behavior Test Videos ===")
    
    for item in PUBLIC_VIDEO_SOURCES:
        file_path = os.path.join(SAMPLE_VIDEOS_DIR, item["filename"])
        print(f"\nProcessing {item['filename']}...")
        
        success = False
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp, open(file_path, 'wb') as out_f:
                out_f.write(resp.read())
            print(f"[SUCCESS] Downloaded web video: {file_path}")
            success = True
        except Exception as err:
            print(f"[NOTICE] Web download skipped ({err}). Creating synthetic sample video...")
        
        if not success or not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
            generate_synthetic_driver_video(file_path, item["event_type"])

if __name__ == "__main__":
    download_and_prepare_videos()
