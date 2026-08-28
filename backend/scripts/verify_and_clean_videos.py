import os
import cv2

TARGET_DIR = "backend/data/sample_videos"

def check_video(file_path):
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    size = os.path.getsize(file_path)
    if size < 1000:
        return False, f"File too small ({size} bytes, likely HTML error page)"
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, "OpenCV failed to open video file"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None or frame.size == 0:
        return False, f"Could not read first frame (Count: {frame_count}, FPS: {fps})"
    
    return True, f"OK ({w}x{h}, {frame_count} frames, {fps:.1f} FPS, {size//1024} KB)"

def main():
    print("=== Checking All Videos in backend/data/sample_videos ===")
    files = [f for f in os.listdir(TARGET_DIR) if f.endswith(('.mp4', '.ogv', '.avi', '.webm'))]
    
    valid_count = 0
    corrupt_files = []
    
    for f in sorted(files):
        path = os.path.join(TARGET_DIR, f)
        is_ok, msg = check_video(path)
        if is_ok:
            print(f"[VALID] {f:<30} -> {msg}")
            valid_count += 1
        else:
            print(f"[INVALID] {f:<30} -> {msg}")
            corrupt_files.append(path)
            
    print(f"\nSummary: {valid_count} valid, {len(corrupt_files)} invalid.")
    
    if corrupt_files:
        print("\nCleaning up invalid/corrupt files...")
        for p in corrupt_files:
            try:
                os.remove(p)
                print(f"[DELETED] {p}")
            except Exception as e:
                print(f"[ERROR] Could not delete {p}: {e}")

if __name__ == "__main__":
    main()
