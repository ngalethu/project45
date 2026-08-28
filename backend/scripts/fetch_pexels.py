import requests
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.pexels.com/"
}

# Working Pexels video CDN links of real drivers
PEXELS_MAP = {
    "pexels_driver_phone.mp4": "https://videos.pexels.com/video-files/5927823/5927823-sd_640_360_25fps.mp4",
    "pexels_driver_seatbelt.mp4": "https://videos.pexels.com/video-files/3052843/3052843-sd_640_360_30fps.mp4",
    "pexels_driver_smoking.mp4": "https://videos.pexels.com/video-files/6863806/6863806-sd_640_360_25fps.mp4",
    "pexels_driver_normal.mp4": "https://videos.pexels.com/video-files/4487361/4487361-sd_640_360_25fps.mp4"
}

out_dir = "backend/data/sample_videos"
os.makedirs(out_dir, exist_ok=True)

for name, url in PEXELS_MAP.items():
    dest = os.path.join(out_dir, name)
    print(f"Fetching {name} from {url}...")
    try:
        r = requests.get(url, headers=headers, timeout=25, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*64):
                    if chunk:
                        f.write(chunk)
            print(f"[SUCCESS] Saved {dest} ({os.path.getsize(dest)} bytes)")
        else:
            print(f"[FAILED] HTTP {r.status_code} for {url}")
    except Exception as e:
        print(f"[ERROR] {e}")
