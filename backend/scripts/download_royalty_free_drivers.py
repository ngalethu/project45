import os
import urllib.request
import json
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Royalty-free high quality real-driver MP4 video clips
ROYALTY_FREE_VIDEOS = [
    {
        "name": "driver_phone_real_1.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-man-driving-a-car-and-talking-on-his-phone-41472-large.mp4"
    },
    {
        "name": "driver_phone_real_2.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-woman-talking-on-the-phone-while-driving-41470-large.mp4"
    },
    {
        "name": "driver_seatbelt_real_1.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-man-fastening-his-seatbelt-in-a-car-41468-large.mp4"
    },
    {
        "name": "driver_drowsy_real_1.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-tired-man-driving-a-car-at-night-41474-large.mp4"
    },
    {
        "name": "driver_steering_real_1.mp4",
        "url": "https://assets.mixkit.co/videos/preview/mixkit-hands-of-a-man-on-the-steering-wheel-of-a-41466-large.mp4"
    }
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
    print("=== Downloading Real-Human Driver Behavior Video Collection ===")
    
    count = 0
    for item in ROYALTY_FREE_VIDEOS:
        dest_path = os.path.join(OUTPUT_DIR, item["name"])
        print(f"\nFetching {item['name']} from: {item['url']}...")
        
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_f:
                while True:
                    chunk = resp.read(64*1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    
            if check_video_validity(dest_path):
                fsz = os.path.getsize(dest_path)
                count += 1
                print(f"[SUCCESS] Downloaded & Validated: {item['name']} ({fsz//1024} KB)")
            else:
                print(f"[INVALID] Deleted {dest_path}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
        except Exception as err:
            print(f"[ERROR] Could not download {item['name']}: {err}")

    print(f"\nDownloaded {count}/{len(ROYALTY_FREE_VIDEOS)} real driver behavior videos successfully.")

if __name__ == "__main__":
    main()
