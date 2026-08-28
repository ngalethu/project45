from __future__ import annotations
import os
import cv2
import numpy as np
import requests
from datetime import datetime, timedelta
import random

SERVER_URL = "http://127.0.0.1:8000"

def create_sample_image(event_type: str, idx: int) -> str:
    os.makedirs("outputs/temp_media", exist_ok=True)
    img_path = f"outputs/temp_media/sample_frame_{idx}.jpg"
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background dashboard mockup
    cv2.rectangle(img, (50, 50), (590, 430), (30, 41, 59), -1) # Dark card
    cv2.circle(img, (320, 200), 80, (71, 85, 105), -1) # Head/driver silhouette
    cv2.rectangle(img, (260, 280), (380, 420), (51, 65, 85), -1) # Body
    
    if event_type == "using_phone":
        # Draw phone box (Red/Orange)
        cv2.rectangle(img, (370, 180), (430, 270), (0, 0, 239), 3)
        cv2.putText(img, "using_phone 0.89", (370, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 239), 2)
        # Phone object
        cv2.rectangle(img, (380, 190), (420, 260), (220, 220, 220), -1)
    elif event_type == "smoking":
        # Draw smoking box (Purple)
        cv2.rectangle(img, (300, 210), (360, 250), (168, 85, 247), 3)
        cv2.putText(img, "smoking 0.84", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (168, 85, 247), 2)
    elif event_type == "no_seatbelt":
        # Draw no_seatbelt box (Yellow)
        cv2.rectangle(img, (240, 270), (400, 410), (0, 215, 255), 3)
        cv2.putText(img, "no_seatbelt 0.92", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    cv2.putText(img, f"DRIVER ALERT: {event_type.upper()}", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Time: {datetime.now().strftime('%H:%M:%S')}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
    
    cv2.imwrite(img_path, img)
    return img_path

def create_sample_video(event_type: str, idx: int) -> str:
    os.makedirs("outputs/temp_media", exist_ok=True)
    video_path = f"outputs/temp_media/sample_clip_{idx}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    
    for f in range(40):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (590, 430), (30, 41, 59), -1)
        cv2.circle(img, (320, 200 + int(np.sin(f/5)*5)), 80, (71, 85, 105), -1)
        cv2.rectangle(img, (260, 280), (380, 420), (51, 65, 85), -1)
        
        offset = int(np.sin(f / 4) * 8)
        if event_type == "using_phone":
            cv2.rectangle(img, (370 + offset, 180), (430 + offset, 270), (0, 0, 239), 3)
            cv2.putText(img, f"using_phone (Frame {f+1})", (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 239), 2)
        elif event_type == "smoking":
            cv2.rectangle(img, (300, 210 + offset), (360, 250 + offset), (168, 85, 247), 3)
            cv2.putText(img, f"smoking (Frame {f+1})", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (168, 85, 247), 2)
        else:
            cv2.rectangle(img, (240, 270), (400, 410), (0, 215, 255), 3)
            cv2.putText(img, f"no_seatbelt (Frame {f+1})", (240, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        out.write(img)
    out.release()
    return video_path

def main():
    events = ["using_phone", "smoking", "no_seatbelt", "using_phone", "smoking"]
    print("Sending alerts with media to cloud server...")

    for i, et in enumerate(events):
        img_p = create_sample_image(et, i+1)
        vid_p = create_sample_video(et, i+1)

        payload = {
            "event_type": et,
            "timestamp": (datetime.now() - timedelta(minutes=i*3)).isoformat(timespec="seconds"),
            "confidence": str(round(random.uniform(0.75, 0.96), 3)),
            "frame_index": str(random.randint(120, 3000)),
            "source_device": f"edge-cab-0{random.randint(1,3)}",
            "notes": f"Demo alert with video clip {i+1}",
        }

        with open(img_p, "rb") as f_img, open(vid_p, "rb") as f_vid:
            files = {
                "frame_file": ("frame.jpg", f_img, "image/jpeg"),
                "clip_file": ("clip.mp4", f_vid, "video/mp4"),
                "raw_clip_file": ("raw_clip.mp4", f_vid, "video/mp4"),
            }
            r = requests.post(f"{SERVER_URL}/alerts", data=payload, files=files, timeout=15)
            r.raise_for_status()
            res = r.json()
            print(f"[OK] Created alert #{res['id']}: {et} | Frame URL: {res.get('frame_path')} | Clip URL: {res.get('clip_path')}")

    print("Successfully seeded 5 media-rich alerts.")

if __name__ == "__main__":
    main()
