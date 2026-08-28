from __future__ import annotations
import os
import re
import urllib.request
import urllib.parse
import json

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Direct high quality Pexels royalty free MP4 video CDN links for real drivers
PEXELS_VIDEOS = [
    {
        "name": "pexels_driver_phone.mp4",
        "search_term": "driver talking on phone",
        "fallback_url": "https://videos.pexels.com/video-files/5927823/5927823-sd_640_360_25fps.mp4"
    },
    {
        "name": "pexels_driver_seatbelt.mp4",
        "search_term": "person driving car seatbelt",
        "fallback_url": "https://videos.pexels.com/video-files/3052843/3052843-sd_640_360_30fps.mp4"
    },
    {
        "name": "pexels_driver_smoking.mp4",
        "search_term": "person smoking in car",
        "fallback_url": "https://videos.pexels.com/video-files/6863806/6863806-sd_640_360_25fps.mp4"
    },
    {
        "name": "pexels_driver_normal.mp4",
        "search_term": "man driving car highway",
        "fallback_url": "https://videos.pexels.com/video-files/4487361/4487361-sd_640_360_25fps.mp4"
    }
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
}

def get_pexels_mp4_from_search(query: str) -> str | None:
    try:
        url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(query)}/"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            # Extract video files matching pexels CDN mp4 patterns
            mp4_urls = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-sd_[0-9_]+fps\.mp4', html)
            if not mp4_urls:
                mp4_urls = re.findall(r'https://videos\.pexels\.com/video-files/[^\s"\'<>]+?\.mp4', html)
            if mp4_urls:
                return mp4_urls[0]
    except Exception as e:
        print(f"Scraping Pexels for '{query}' failed: {e}")
    return None

def download_file(url: str, dest_path: str):
    print(f"Downloading from: {url}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, 'wb') as out_f:
        out_f.write(resp.read())
    size = os.path.getsize(dest_path)
    print(f"[SUCCESS] Downloaded {dest_path} ({size} bytes)")

def main():
    print("=== Downloading Real Human Driver Videos from Pexels ===")
    for item in PEXELS_VIDEOS:
        dest_path = os.path.join(OUTPUT_DIR, item["name"])
        print(f"\n---> Preparing video: {item['name']}...")
        
        mp4_url = get_pexels_mp4_from_search(item["search_term"])
        if not mp4_url:
            mp4_url = item["fallback_url"]
            print(f"Using direct Pexels CDN URL...")
        
        try:
            download_file(mp4_url, dest_path)
        except Exception as err:
            print(f"Failed download from primary URL ({err}), trying fallback CDN...")
            try:
                download_file(item["fallback_url"], dest_path)
            except Exception as err2:
                print(f"ERROR downloading {item['name']}: {err2}")

if __name__ == "__main__":
    main()
