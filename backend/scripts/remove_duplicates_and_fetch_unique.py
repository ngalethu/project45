import os
import hashlib
import cv2

TARGET_DIR = "backend/data/sample_videos"

def get_file_hash(path: str) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def deduplicate_videos():
    print("=== Checking & Deduplicating Video Collection ===")
    files = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith(".mp4")])
    
    seen_hashes = {}
    duplicates_removed = 0
    unique_files = []
    
    for f in files:
        path = os.path.join(TARGET_DIR, f)
        h = get_file_hash(path)
        
        if h in seen_hashes:
            print(f"[DUPLICATE] Removing '{f}' (Identical content to '{seen_hashes[h]}')")
            try:
                os.remove(path)
                duplicates_removed += 1
            except Exception as e:
                print(f"[ERROR] Could not delete {f}: {e}")
        else:
            seen_hashes[h] = f
            unique_files.append(f)
            
    print(f"\nDeduplication Complete: Removed {duplicates_removed} duplicate files.")
    print(f"Total Remaining Unique Videos: {len(unique_files)}\n")
    
    for idx, u in enumerate(unique_files, 1):
        p = os.path.join(TARGET_DIR, u)
        cap = cv2.VideoCapture(p)
        w, h, count, fps = 0, 0, 0, 0
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        size_kb = os.path.getsize(p) // 1024
        print(f"{idx:2d}. {u:<38} -> {w}x{h} | {count} frames | {fps:.1f} FPS | {size_kb} KB")

if __name__ == "__main__":
    deduplicate_videos()
