from __future__ import annotations
import os
import re
import urllib.request
import urllib.parse
import json

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keyword list requested by user on Pexels
SEARCH_TOPICS = [
    {"name": "pexels_driver_phone.mp4", "query": "driver talking on phone"},
    {"name": "pexels_driver_seatbelt.mp4", "query": "fasten seatbelt car"},
    {"name": "pexels_driver_smoking.mp4", "query": "driver smoking car"},
    {"name": "pexels_driver_drowsy.mp4", "query": "drowsy driver car"}
]

# Standard Chrome headers to crawl without GUI browser popup
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}

def crawl_pexels_video_link(query: str) -> str | None:
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.pexels.com/search/videos/{encoded_query}/"
    print(f"[CRAWLER] Crawling Pexels search page for: '{query}' ({search_url})...")
    
    try:
        req = urllib.request.Request(search_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            print(f"[CRAWLER] Received HTML page (Length: {len(html)} bytes)")
            
            # Find video file links in Pexels HTML JSON / srcset data
            matches = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
            if not matches:
                matches = re.findall(r'https://images\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
            if not matches:
                matches = re.findall(r'https://[^\s"\'<>]+?pexels[^\s"\'<>]+?\.mp4', html)
                
            if matches:
                unique_urls = list(dict.fromkeys(matches))
                print(f"[CRAWLER] Found {len(unique_urls)} video MP4 links!")
                # Prefer 640_360 or 720p sd videos for fast download
                sd_urls = [u for u in unique_urls if "360" in u or "720" in u or "sd" in u]
                selected_url = sd_urls[0] if sd_urls else unique_urls[0]
                return selected_url
            else:
                print(f"[CRAWLER] No direct MP4 URL found in HTML regex for '{query}'")
    except Exception as err:
        print(f"[CRAWLER ERROR] Failed to fetch search page for '{query}': {err}")
    return None

def download_video_stream(url: str, output_path: str) -> bool:
    print(f"[DOWNLOAD] Stream fetching from URL: {url} -> {output_path}")
    download_headers = HEADERS.copy()
    download_headers["Referer"] = "https://www.pexels.com/"
    
    try:
        req = urllib.request.Request(url, headers=download_headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, "wb") as f:
            total_bytes = 0
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
        print(f"[SUCCESS] Download completed! Saved file: {output_path} ({total_bytes} bytes)")
        return True
    except Exception as e:
        print(f"[DOWNLOAD ERROR] Failed downloading from {url}: {e}")
        return False

def main():
    print("==================================================")
    print("PEXELS HEADLESS VIDEO CRAWLER (NO BROWSER GUI)")
    print("==================================================")
    
    downloaded_files = []
    
    for item in SEARCH_TOPICS:
        target_name = item["name"]
        query = item["query"]
        output_file = os.path.join(OUTPUT_DIR, target_name)
        
        print(f"\n---> Searching & Crawling Topic: {query}")
        video_url = crawl_pexels_video_link(query)
        
        if video_url:
            success = download_video_stream(video_url, output_file)
            if success:
                downloaded_files.append((target_name, output_file, video_url))
        else:
            print(f"[WARNING] Could not obtain video URL for '{query}'")
            
    print("\n==================================================")
    print(f"CRAWLING SUMMARY: Successfully downloaded {len(downloaded_files)}/{len(SEARCH_TOPICS)} videos")
    for name, path, url in downloaded_files:
        print(f"- {name}: {os.path.getsize(path)} bytes | URL: {url}")
    print("==================================================")

if __name__ == "__main__":
    main()
