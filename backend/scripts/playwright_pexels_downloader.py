import os
import re
import urllib.request
import urllib.parse
import cv2
from playwright.sync_api import sync_playwright

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUERIES = [
    {"name": "pexels_phone_1.mp4", "query": "driver talking on phone"},
    {"name": "pexels_phone_2.mp4", "query": "person texting while driving"},
    {"name": "pexels_seatbelt_1.mp4", "query": "fasten seatbelt car"},
    {"name": "pexels_seatbelt_2.mp4", "query": "driver wearing seatbelt"},
    {"name": "pexels_smoking_1.mp4", "query": "driver smoking in car"},
    {"name": "pexels_smoking_2.mp4", "query": "person smoking cigarette driving"},
    {"name": "pexels_drowsy_1.mp4", "query": "drowsy driver yawning"},
    {"name": "pexels_drowsy_2.mp4", "query": "tired sleeping driver car"},
    {"name": "pexels_driving_1.mp4", "query": "man driving car highway"},
    {"name": "pexels_driving_2.mp4", "query": "woman driving car city"}
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
    print("PLAYWRIGHT HEADLESS PEXELS VIDEO CRAWLER")
    print(" (Runs 100% in background - NO browser GUI window) ")
    print("==================================================")
    
    with sync_playwright() as p:
        # Launch browser 100% headless
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()
        
        success_count = 0
        
        for idx, item in enumerate(QUERIES, 1):
            out_name = item["name"]
            query = item["query"]
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            if os.path.exists(out_path) and check_video_validity(out_path):
                print(f"[{idx}/{len(QUERIES)}] Video already exists & valid: {out_name}")
                success_count += 1
                continue
                
            search_url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(query)}/"
            print(f"\n[{idx}/{len(QUERIES)}] Navigating headlessly to: {query}...")
            
            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
                page.wait_for_timeout(2000)
                
                # Extract all video source mp4 links from DOM
                video_elements = page.query_selector_all("source[src*='.mp4'], video source, a[href*='.mp4']")
                mp4_urls = []
                for el in video_elements:
                    src = el.get_attribute("src") or el.get_attribute("href")
                    if src and ".mp4" in src:
                        mp4_urls.append(src)
                        
                if not mp4_urls:
                    # Search page HTML content if DOM query returned empty
                    content = page.content()
                    mp4_urls = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', content)
                    
                unique_urls = list(dict.fromkeys(mp4_urls))
                print(f"[FOUND] {len(unique_urls)} MP4 URLs for '{query}'")
                
                if unique_urls:
                    # Pick 360p / 720p sd link for quick download
                    selected_url = unique_urls[0]
                    for u in unique_urls:
                        if "360" in u or "720" in u or "sd" in u:
                            selected_url = u
                            break
                            
                    print(f"Downloading: {selected_url} -> {out_name}...")
                    
                    # Fetch stream via Playwright request API
                    resp = context.request.get(selected_url, headers={"Referer": "https://www.pexels.com/"}, timeout=30000)
                    if resp.status == 200:
                        with open(out_path, "wb") as f:
                            f.write(resp.body())
                            
                        if check_video_validity(out_path):
                            fsz = os.path.getsize(out_path)
                            success_count += 1
                            print(f"[SUCCESS] Saved & Validated: {out_name} ({fsz//1024} KB)")
                        else:
                            print(f"[INVALID] Video file failed check: {out_name}")
                            if os.path.exists(out_path):
                                os.remove(out_path)
                    else:
                        print(f"[FAILED] HTTP {resp.status} fetching video stream")
                else:
                    print(f"[WARNING] No MP4 URLs found for '{query}'")
            except Exception as err:
                print(f"[ERROR] Failed crawling '{query}': {err}")
                
        browser.close()
        
    print("\n==================================================")
    print(f"PLAYWRIGHT CRAWLING COMPLETE!")
    print(f"Successfully Collected Videos: {success_count}/{len(QUERIES)}")
    print("==================================================")

if __name__ == "__main__":
    main()
