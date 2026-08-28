import os
import requests

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GITHUB_RAW_VIDEOS = [
    {
        "name": "d3s_driver_yawn.mp4",
        "url": "https://raw.githubusercontent.com/bindujiit/Driver-Drowsiness-Dataset-D3S/main/Sample_Videos/yawn.mp4"
    },
    {
        "name": "d3s_driver_drowsy.mp4",
        "url": "https://raw.githubusercontent.com/bindujiit/Driver-Drowsiness-Dataset-D3S/main/Sample_Videos/drowsy.mp4"
    },
    {
        "name": "wikimedia_driver_distracted.mp4",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Distracted_driver_talking_on_cell_phone.ogv"
    },
    {
        "name": "github_driver_activity.mp4",
        "url": "https://raw.githubusercontent.com/humza909/Dataset-Video-Driver-Activity-Recognition/master/sample_driver.mp4"
    }
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def main():
    print("=== Downloading Real Driver Videos from GitHub & Open Access Repos ===")
    for item in GITHUB_RAW_VIDEOS:
        dest_path = os.path.join(OUTPUT_DIR, item["name"])
        print(f"\nDownloading {item['name']} from {item['url']}...")
        try:
            r = requests.get(item["url"], headers=headers, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"[SUCCESS] Saved {dest_path} ({len(r.content)} bytes)")
            else:
                print(f"[SKIP] Status code: {r.status_code}, Length: {len(r.content)}")
        except Exception as e:
            print(f"[ERROR] Could not download {item['name']}: {e}")

if __name__ == "__main__":
    main()
