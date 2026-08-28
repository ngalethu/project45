import cloudscraper
import time
import re
import urllib.parse
import os
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUERIES = [
    "driver talking phone",
    "driver using mobile",
    "person driving car seatbelt",
    "fasten seatbelt car",
    "driver smoking cigarette",
    "driver smoking vape",
    "drowsy driver yawning",
    "tired driver sleeping",
    "man driving car highway",
    "woman driving car city",
    "driver distracted phone",
    "night driving car inside"
]

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
    print("==================================================")
    print("CLOUDSCRAPER PEXELS BULK VIDEO CRAWLER (HEADLESS)")
    print("==================================================")
    
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    
    downloaded_count = 0
    total_bytes = 0
    
    for idx, q in enumerate(QUERIES, 1):
        time.sleep(1.0)
        search_url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(q)}/"
        print(f"\n[{idx}/{len(QUERIES)}] Crawling Pexels topic: '{q}'...")
        
        try:
            res = scraper.get(search_url, timeout=20)
            if res.status_code == 200:
                html = res.text
                matches = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
                if not matches:
                    matches = re.findall(r'https://images\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', html)
                
                unique_urls = list(dict.fromkeys(matches))
                sd_urls = [u for u in unique_urls if "360" in u or "720" in u or "sd" in u or "540" in u]
                final_urls = sd_urls if sd_urls else unique_urls
                
                print(f"[FOUND] {len(final_urls)} video MP4 links for '{q}'.")
                
                # Download up to 2 videos per topic
                for v_i, video_url in enumerate(final_urls[:2], 1):
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', q).strip('_')
                    out_name = f"pexels_{clean_name}_{v_i}.mp4"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    
                    if os.path.exists(out_path) and check_video_validity(out_path):
                        print(f"[SKIP] Valid video already exists: {out_name}")
                        downloaded_count += 1
                        continue
                        
                    print(f"Downloading #{v_i}: {out_name}...")
                    v_res = scraper.get(video_url, timeout=35, stream=True)
                    if v_res.status_code == 200:
                        with open(out_path, "wb") as f:
                            for chunk in v_res.iter_content(chunk_size=64*1024):
                                if chunk:
                                    f.write(chunk)
                                    
                        if check_video_validity(out_path):
                            fsz = os.path.getsize(out_path)
                            total_bytes += fsz
                            downloaded_count += 1
                            print(f"[SUCCESS] Downloaded & Validated: {out_name} ({fsz//1024} KB)")
                        else:
                            print(f"[INVALID] Removed corrupted file: {out_name}")
                            if os.path.exists(out_path):
                                os.remove(out_path)
                    else:
                        print(f"[FAIL] HTTP {v_res.status_code} for video link")
            else:
                print(f"[HTTP {res.status_code}] Failed to fetch Pexels search page for '{q}'")
        except Exception as err:
            print(f"[ERROR] {q}: {err}")
            
    print("\n==================================================")
    print(f"BULK CRAWL FINISHED!")
    print(f"Total Valid Videos in Collection: {downloaded_count}")
    print(f"Total Download Size: {total_bytes // (1024*1024)} MB")
    print("==================================================")

if __name__ == "__main__":
    main()
