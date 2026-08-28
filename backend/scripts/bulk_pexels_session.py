import requests
import time
import re
import urllib.parse
import os
import cv2

OUTPUT_DIR = "backend/data/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUERIES = [
    "driver phone",
    "driver talking phone",
    "driver mobile",
    "driver seatbelt",
    "car seatbelt",
    "driver smoking",
    "smoking car",
    "drowsy driver",
    "yawning driver",
    "distracted driver",
    "man driving car",
    "woman driving car",
    "night driving car",
    "car steering wheel"
]

session = requests.Session()
session.headers.update({
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
})

# Warm up session on pexels main page
try:
    print("Warming up Pexels session...")
    r = session.get("https://www.pexels.com/", timeout=10)
    print("Session status:", r.status_code)
except Exception as e:
    print("Warmup failed:", e)

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

downloaded_count = 0
for idx, q in enumerate(QUERIES, 1):
    time.sleep(1.5) # Sleep to avoid rate limiting
    url = f"https://www.pexels.com/search/videos/{urllib.parse.quote(q)}/"
    print(f"\n[{idx}/{len(QUERIES)}] Searching Pexels: '{q}'...")
    
    try:
        res = session.get(url, timeout=15)
        print(f"Status: {res.status_code} | Len: {len(res.text)}")
        if res.status_code == 200:
            matches = re.findall(r'https://videos\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', res.text)
            if not matches:
                matches = re.findall(r'https://images\.pexels\.com/video-files/[0-9]+/[0-9]+-[^"\'\s]+\.mp4', res.text)
            
            unique = list(dict.fromkeys(matches))
            print(f"Found {len(unique)} MP4 video links.")
            
            if unique:
                # Pick up to 2 videos per query
                for v_i, video_url in enumerate(unique[:2], 1):
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', q).strip('_')
                    out_path = os.path.join(OUTPUT_DIR, f"pexels_{clean_name}_{v_i}.mp4")
                    
                    if os.path.exists(out_path) and check_video_validity(out_path):
                        print(f"[EXISTS] {out_path}")
                        downloaded_count += 1
                        continue
                        
                    print(f"Downloading #{v_i}: {video_url} -> {out_path}...")
                    dl_headers = session.headers.copy()
                    dl_headers["Referer"] = "https://www.pexels.com/"
                    
                    v_res = session.get(video_url, headers=dl_headers, timeout=30, stream=True)
                    if v_res.status_code == 200:
                        with open(out_path, "wb") as f:
                            for chunk in v_res.iter_content(chunk_size=64*1024):
                                if chunk:
                                    f.write(chunk)
                        if check_video_validity(out_path):
                            downloaded_count += 1
                            print(f"[SUCCESS] Downloaded: {out_path} ({os.path.getsize(out_path)//1024} KB)")
                        else:
                            print(f"[INVALID] Deleted {out_path}")
                            if os.path.exists(out_path):
                                os.remove(out_path)
                    else:
                        print(f"[FAIL] Video HTTP {v_res.status_code}")
    except Exception as err:
        print(f"Error searching '{q}': {err}")

print(f"\nTotal Valid Videos Downloaded: {downloaded_count}")
