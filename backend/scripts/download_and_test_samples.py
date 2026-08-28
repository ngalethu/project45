from __future__ import annotations
import os
import urllib.request
import requests
import cv2
import numpy as np

SERVER_URL = "http://127.0.0.1:8000"

SAMPLE_URLS = [
    {
        "name": "test_driver_phone.jpg",
        "url": "https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=800&q=80",
        "expected": "using_phone"
    },
    {
        "name": "test_driver_smoking.jpg",
        "url": "https://images.unsplash.com/photo-1527061011665-3652c757a4d4?w=800&q=80",
        "expected": "smoking"
    },
    {
        "name": "test_driver_seatbelt.jpg",
        "url": "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800&q=80",
        "expected": "seatbelt/no_seatbelt"
    }
]

def download_samples():
    os.makedirs("data/sample_test_media", exist_ok=True)
    downloaded_paths = []
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for item in SAMPLE_URLS:
        file_path = os.path.join("data/sample_test_media", item["name"])
        try:
            print(f"Downloading sample image: {item['name']}...")
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[SUCCESS] Saved to {file_path}")
            downloaded_paths.append(file_path)
        except Exception as e:
            print(f"[WARNING] Could not download from web ({e}), creating local synthetic sample...")
            # Create synthetic realistic driver sample image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (620, 460), (30, 41, 59), -1)
            cv2.circle(img, (320, 180), 75, (100, 116, 139), -1) # Head
            cv2.rectangle(img, (240, 255), (400, 440), (71, 85, 105), -1) # Body
            
            if "phone" in item["name"]:
                cv2.rectangle(img, (380, 150), (430, 240), (0, 0, 255), -1)
                cv2.putText(img, "USING PHONE DEMO", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif "smoking" in item["name"]:
                cv2.rectangle(img, (310, 210), (360, 225), (200, 200, 255), -1)
                cv2.putText(img, "SMOKING DEMO", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.line(img, (250, 260), (390, 430), (0, 255, 255), 4)
                cv2.putText(img, "NO SEATBELT DEMO", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imwrite(file_path, img)
            downloaded_paths.append(file_path)
            
    return downloaded_paths

def test_auto_detection(file_paths):
    print("\n==================================================")
    print("Testing Auto AI Detection API on Downloaded Media")
    print("==================================================")
    
    for path in file_paths:
        with open(path, "rb") as f:
            files = {"media_file": (os.path.basename(path), f, "image/jpeg")}
            data = {"notes": f"Auto AI detection test for {os.path.basename(path)}"}
            res = requests.post(f"{SERVER_URL}/api/auto_detect_media", files=files, data=data, timeout=15)
            if res.status_code == 200:
                result = res.json()
                alert = result["alert"]
                det = result["detection"]
                print(f"[AI DETECTED] File: {os.path.basename(path)} -> Event: {det['event_type']} | Conf: {det['confidence']} | Bbox URL: {det.get('frame_url')}")
            else:
                print(f"[ERROR] Failed to process {path}: {res.text}")

if __name__ == "__main__":
    paths = download_samples()
    test_auto_detection(paths)
