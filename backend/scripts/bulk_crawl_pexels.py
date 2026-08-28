from __future__ import annotations
import os
import re
import urllib.request
import urllib.parse
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Comprehensive list of queries for driver behavior monitoring
SEARCH_CATEGORIES = [
    "driver talking on phone",
    "person calling while driving",
    "driver texting car",
    "driver smoking cigarette",
    "driver smoking vape",
    "driver wearing seatbelt",
    "fasten seatbelt car",
    "drowsy driver yawning",
    "tired driver sleeping car",
    "man driving car highway",
    "woman driving car city",
    "driver distracted car",
    "steering wheel driving car",
    "driver looking away phone",
    "night driving car inside"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1"
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

def crawl_pexels_mp4_links(query: str, max_videos_per_query: int = 3) -> list[str]:
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.pexels.com/search/videos/{encoded_query}/"
    print(f"\n[SEARCH] Searching Pexels for: '{query}'...")
    
    try:
        req = urllib.request.Request(search_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            
            # Find all video files in Pexels HTML
            matches = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
            if not matches:
                matches = re.findall(r'https://images\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
            if not matches:
                matches = re.findall(r'https://[^\s"\'<>]+?pexels[^\s"\'<>]+?\.mp4', html)
                
            unique_links = list(dict.fromkeys(matches))
            # Prefer 360p / 720p sd videos for quick download
            sd_links = [u for u in unique_links if "360" in u or "720" in u or "sd" in u or "540" in u]
            final_links = sd_links if sd_links else unique_links
            
            print(f"[FOUND] Found {len(final_links)} candidate MP4 video URLs.")
            return final_links[:max_videos_per_query]
    except Exception as err:
        print(f"[SEARCH ERROR] {query}: {err}")
        return []

def download_file(url: str, output_path: str) -> bool:
    dl_headers = HEADERS.copy()
    dl_headers["Referer"] = "https://www.pexels.com/"
    
    try:
        req = urllib.request.Request(url, headers=dl_headers)
        with urllib.request.urlopen(req, timeout=25) as resp, open(output_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
        return check_video_validity(output_path)
    except Exception as e:
        print(f"[DOWNLOAD FAILED] {url}: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        return False

def main():
    print("==================================================")
    print("BULK PEXELS VIDEO CRAWLER & DOWNLOADER (HEADLESS)")
    print("==================================================")
    
    downloaded_count = 0
    total_downloaded_bytes = 0
    
    for cat_idx, category in enumerate(SEARCH_CATEGORIES, 1):
        mp4_urls = crawl_pexels_mp4_links(category, max_videos_per_query=2)
        
        for v_idx, url in enumerate(mp4_urls, 1):
            clean_cat_name = re.sub(r'[^a-zA-Z0-9]', '_', category).strip('_')
            filename = f"pexels_{clean_cat_name}_{v_idx}.mp4"
            dest_path = os.path.join(OUTPUT_DIR, filename)
            
            if os.path.exists(dest_path) and check_video_validity(dest_path):
                print(f"[SKIP] File already exists and is valid: {filename}")
                downloaded_count += 1
                continue
                
            print(f"[{cat_idx}/{len(SEARCH_CATEGORIES)}] Downloading #{v_idx} for '{category}' -> {filename}...")
            success = download_file(url, dest_path)
            if success:
                fsize = os.path.getsize(dest_path)
                total_downloaded_bytes += fsize
                downloaded_count += 1
                print(f"[SUCCESS] Downloaded & Validated: {filename} ({fsize//1024} KB)")
            else:
                print(f"[WARNING] Skipping invalid file download for {filename}")

    print("\n==================================================")
    print(f"BULK CRAWL COMPLETED!")
    print(f"Total Valid Videos: {downloaded_count}")
    print(f"Total Size Downloaded: {total_downloaded_bytes // (1024*1024)} MB")
    print("==================================================")

if __name__ == "__main__":
    main()
