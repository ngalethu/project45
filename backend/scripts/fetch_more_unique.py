import os
import urllib.request
import hashlib
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NEW_UNIQUE_VIDEOS = [
    {
        "name": "mixkit_driver_phone_call.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-man-driving-a-car-and-talking-on-his-phone-41472-large.mp4"
    },
    {
        "name": "mixkit_driver_fasten_seatbelt.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-man-fastening-his-seatbelt-in-a-car-41468-large.mp4"
    },
    {
        "name": "mixkit_driver_night_tired.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-tired-man-driving-a-car-at-night-41474-large.mp4"
    },
    {
        "name": "mixkit_driver_steering_wheel.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-hands-of-a-man-on-the-steering-wheel-of-a-41466-large.mp4"
    }
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def check_video_validity(file_path: str) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 5000:
        return False
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None and frame.size > 0
    except Exception:
        return False

def main():
    print("=== Downloading Additional Unique Driver Behavior Videos ===")
    
    for item in NEW_UNIQUE_VIDEOS:
        dest_path = os.path.join(OUTPUT_DIR, item["name"])
        print(f"Downloading {item['name']}...")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if check_video_validity(dest_path):
                print(f"[SUCCESS] Saved {item['name']} ({os.path.getsize(dest_path)//1024} KB)")
            else:
                print(f"[INVALID] Removed {item['name']}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
        except Exception as e:
            print(f"[ERROR] Downloading {item['name']}: {e}")

if __name__ == "__main__":
    main()
