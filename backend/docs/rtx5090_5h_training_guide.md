# Chạy DMS YOLO11m trên máy thuê RTX 5090 trong 5–6 giờ

Notebook: `backend/driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb`.

Sau khi chép dự án sang máy RTX 5090, có thể chạy preflight và mở notebook bằng một lệnh:

```powershell
powershell -ExecutionPolicy Bypass -File .\backend\scripts\start_rtx5090_training.ps1 -LaunchJupyter
```

Script dừng trước khi mở Jupyter nếu GPU không phải RTX 5090, PyTorch thiếu `sm_120`, dataset/model bị thiếu hoặc Drive checkpoint chưa truy cập được.

## 1. Kiểm tra PyTorch Blackwell

RTX 5090 dùng compute capability `sm_120`, cần PyTorch được build với CUDA 12.8 trở lên. Trong terminal Windows của máy thuê:

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install --upgrade "ultralytics>=8.3,<9" google-api-python-client google-auth pyyaml
```

Khởi động lại Jupyter kernel sau khi cài. Cell đầu notebook phải in `GPU capability: (12, 0)`, danh sách kiến trúc có `sm_120` và `CUDA matmul probe: OK`.

## 2. Đặt dự án và dữ liệu

Chép nguyên dự án sang ổ SSD/NVMe của máy thuê, không train trực tiếp từ ổ mạng chậm. Giữ cấu trúc:

```text
project4/
  backend/driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb
  backend/yolo11m.pt
  data/processed/dms_yolo_3class_v3_curated/dms_dataset.yaml
```

Nếu vị trí khác, mở PowerShell dùng để khởi động Jupyter và đặt:

```powershell
$env:DMS_PROJECT_ROOT = "D:\project4"
$env:DMS_RUNS_ROOT = "D:\dms_runs"
```

Nếu `nas1` là vùng lưu trữ còn tồn tại sau khi máy thuê hết hạn, có thể đặt `DMS_RUNS_ROOT` vào thư mục trên NAS. Chỉ làm vậy khi tốc độ NAS đủ nhanh; dữ liệu train vẫn nên ở SSD local.

## 3. Checkpoint Google Drive

Khuyến nghị cài rclone và cung cấp remote `gdrive:`. Notebook v9 dùng rclone trên cả Kaggle và Windows local, kiểm tra ghi/đọc trước khi train và upload checkpoint sau từng epoch.

Nếu đã có cấu hình base64, đặt nó chỉ trong phiên PowerShell dùng để mở Jupyter:

```powershell
$env:RCLONE_CONFIG_B64 = (Get-Clipboard -Raw).Trim()
$env:DMS_REQUIRE_REMOTE_CHECKPOINT = "1"
powershell -ExecutionPolicy Bypass -File .\backend\scripts\start_rtx5090_training.ps1 -LaunchJupyter
```

Hãy copy secret base64 vào clipboard của máy thuê trước khi chạy. Script chỉ tạo `rclone.conf` tạm để kiểm tra rồi xóa; notebook tự tạo file tạm riêng trong phiên train.

Không lưu giá trị secret vào notebook, GitHub hoặc ảnh chụp màn hình. Chỉ train sau khi thấy `[CHECKPOINT READY]`.

Nếu dùng NAS làm vùng persistent và không muốn Drive, đặt:

```powershell
$env:DMS_RUNS_ROOT = "<DUONG_DAN_NAS_PERSISTENT>"
$env:DMS_REQUIRE_REMOTE_CHECKPOINT = "0"
```

## 4. Ngân sách 5,5 giờ

Mặc định notebook v9 dùng:

```text
80 epoch base, tối đa 4.5 giờ
20 epoch fine-tune, tối đa 0.5 giờ
0.5 giờ dự phòng test, ONNX export và upload
batch=32, workers=4 trên RTX 5090
```

Có thể đổi mà không sửa notebook:

```powershell
$env:DMS_TOTAL_BUDGET_HOURS = "5.5"
$env:DMS_BASE_TRAIN_HOURS = "4.5"
$env:DMS_FINE_TRAIN_HOURS = "0.5"
```

Nếu CUDA out-of-memory, đặt `BATCH = 24` trong cell cấu hình. Nếu DataLoader trên Windows lỗi multiprocessing, đặt `WORKERS = 0`.

## 5. Dấu hiệu an toàn

Trước train:

```text
CUDA matmul probe: OK
[RCLONE VERIFIED] gdrive:
[CHECKPOINT READY] Đã kiểm tra ghi và đọc metadata thành công.
```

Sau epoch đầu:

```text
[RCLONE OK] yolo11m_dms_3class_v3_base/epoch_001.pt
```

Khi hết thời gian, tải `champion_artifacts/best.pt`, `best.onnx` và `metrics_summary.json`, hoặc lấy các file đã đồng bộ trên Google Drive.
