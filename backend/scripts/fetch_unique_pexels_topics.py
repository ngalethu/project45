import cloudscraper
import hashlib
import time
import re
import urllib.parse
import os
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNIQUE_TOPICS = [
    ("unique_pexels_driver_texting.mp4", "driver texting while driving"),
    ("unique_pexels_driver_seatbelt.mp4", "fastening car seatbelt"),
    ("unique_pexels_driver_yawn.mp4", "driver yawning tired"),
    ("unique_pexels_driver_coffee.mp4", "driver drinking coffee car"),
    ("unique_pexels_driver_steering.mp4", "hands on steering wheel driving")
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

def get_existing_hashes():
    hashes = set()
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp4"):
            p = os.path.join(OUTPUT_DIR, f)
            try:
                h = hashlib.md5(open(p, "rb").read()).hexdigest()
                hashes.add(h)
            except Exception:
                pass
    return hashes

def main():
    print("=== Crawling Additional Unique Video Topics ===")
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    
    existing_hashes = get_existing_hashes()
    added_count = 0
    
    for filename, query in UNIQUE_TOPICS:
        time.sleep(1.2)
        search_url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(query)}/"
        print(f"\nSearching Pexels for: '{query}'...")
        
        try:
            res = scraper.get(search_url, timeout=20)
            if res.status_code == 200:
                matches = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', res.text)
                unique_urls = list(dict.fromkeys(matches))
                
                for candidate_url in unique_urls:
                    dest_path = os.path.join(OUTPUT_DIR, filename)
                    v_res = scraper.get(candidate_url, timeout=30, stream=True)
                    if v_res.status_code == 200:
                        with open(dest_path, "wb") as f:
                            for chunk in v_res.iter_content(64*1024):
                                if chunk:
                                    f.write(chunk)
                                    
                        if check_video_validity(dest_path):
                            cand_hash = hashlib.md5(open(dest_path, "rb").read()).hexdigest()
                            if cand_hash not in existing_hashes:
                                existing_hashes.add(cand_hash)
                                added_count += 1
                                print(f"[SUCCESS] Added UNIQUE video: {filename} ({os.path.getsize(dest_path)//1024} KB)")
                                break
                            else:
                                print(f"[DUPLICATE HASH] Skipping candidate video for '{query}'...")
                                if os.path.exists(dest_path):
                                    os.remove(dest_path)
        except Exception as e:
            print(f"[ERROR] '{query}': {e}")
            
    print(f"\nFinished: Added {added_count} brand-new unique videos.")

if __name__ == "__main__":
    main()
