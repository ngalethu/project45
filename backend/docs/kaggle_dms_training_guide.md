# Hướng dẫn train DMS YOLO11 trên Kaggle Web

## Trạng thái dữ liệu thực tế

- Detection hợp lệ: 38.969 ảnh, 38.969 label.
- Split group-disjoint: 31.324 train, 3.833 validation, 3.812 test.
- Canonical classes: `phone`, `seatbelt`, `no-seatbelt`, `smoking`.
- AUC v2 local: 32.714 entry classification nhưng ZIP đang encrypted; chỉ dùng 5.418 phone-candidate khi có mật khẩu/quyền truy cập hợp lệ.
- Seatbelt Real tải được: 8 ảnh chưa có bounding box, được xét pseudo-label.
- Một ảnh raw 0 byte và label đi kèm đã bị xóa theo yêu cầu.
- Audit phát hiện 3.004 nhóm từng nằm chéo các split nguồn; dataset v2 đã chia lại theo group.

Không cam kết trước mAP/F1 >85%. Chỉ kết luận đạt khi file `metrics_summary.json` của test set có `target_met: true`.

## File sử dụng

- Notebook: `backend/driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb`
- Script kiểm tra Drive local/rclone: `backend/scripts/setup_checkpoint_drive.ps1`
- Bundle upload: `backend/outputs/kaggle_dms_bundle/`
- Dataset YAML local: `data/processed/dms_yolo_4class_v2/dms_dataset.yaml`
- Audit: `data/processed/dms_yolo_4class_v2/audit_report.json`

## Upload bằng giao diện web

1. Mở <https://www.kaggle.com/datasets> và chọn **New Dataset**.
2. Đặt dataset ở chế độ **Private** trong lúc kiểm tra license.
3. Upload toàn bộ file trong `backend/outputs/kaggle_dms_bundle/`.
4. Sau khi dataset xử lý xong, mở <https://www.kaggle.com/code> → **New Notebook**.
5. Chọn **File → Import Notebook**, upload file `.ipynb` trong bundle.
6. Chọn **Add Input** và attach dataset vừa upload.
7. Trong **Settings → Accelerator**, chọn GPU (ưu tiên P100 hoặc T4).
8. Cấu hình một backend checkpoint thật theo mục bên dưới; cell `checkpoint_sync_ready.json` phải chạy thành công.
9. Chạy lần lượt các cell kiểm tra môi trường/dataset trước. Khi audit đúng, chọn **Run All**.
10. Cuối phiên, chọn **Save Version → Save & Run All**.
11. Tải `dms_champion_artifacts.zip` ở tab **Output**. File chứa `best.pt`, `best.onnx` và `metrics_summary.json`.

`/kaggle/working` chỉ là scratch disk. Notebook lưu `epoch_NNN.pt` sau từng epoch vào Google Drive, xác minh size/MD5 và tự tải epoch mới nhất để resume ở phiên sau. Không đặt một thư mục tên `drive` trong Kaggle output rồi coi đó là Google Drive.

## Checkpoint Google Drive

Folder đích: <https://drive.google.com/drive/folders/1RfDV984zjw0Y5yfnxtnd7pPQhJpNczt_>.

### Cách 1 — Google Drive API

1. Bật Google Drive API trong Google Cloud Console và tạo OAuth Client loại **Desktop app**.
2. Lấy refresh token của chính tài khoản có quyền Editor trên folder.
3. Tạo Kaggle Secret `GDRIVE_OAUTH_JSON` theo mẫu trong notebook và bật secret cho notebook.
4. Không dùng service account với My Drive cá nhân; chỉ dùng nó khi folder nằm trong Shared Drive và đã cấp quyền phù hợp.

### Cách 2 — rclone dự phòng

Trên máy Windows, chạy:

```powershell
& "D:\.idea\project4\backend\scripts\setup_checkpoint_drive.ps1" -CopyKaggleSecret
```

Script kiểm tra remote `gdrive`, folder ID và `H:\My Drive\project3_runs`, sau đó copy cấu hình base64 vào clipboard mà không in token. Dán giá trị clipboard vào Kaggle Secret `RCLONE_CONFIG_B64`. Khi Internet được bật, notebook chỉ cài `rclone` bằng apt nếu secret này thực sự tồn tại.

Notebook gọi remote thuần `gdrive:` và truyền folder đích qua biến backend chính thức `RCLONE_DRIVE_ROOT_FOLDER_ID`. Cách này tương thích với rclone cũ trong Kaggle và tránh lỗi `config name contains invalid characters` do connection-string chứa dấu phẩy.

Notebook ưu tiên Drive API nếu cả hai cách đều được cấu hình. Nếu không xác minh được backend thật, notebook dừng trước cell train.

### Thiết lập ổn định cho Kaggle

- Mặc định `BATCH=8`, `WORKERS=2` để giảm lỗi CUDA OOM và DataLoader worker trên GPU T4/P100.
- Notebook yêu cầu còn ít nhất 8 GiB trong `/kaggle/working` trước khi giải nén và train.
- Ultralytics được giữ trong dải `>=8.3,<9`; bản Kaggle quá cũ hoặc từ major version khác sẽ được thay bằng bản tương thích.
- Base detector và pseudo fine-tune có checkpoint/resume riêng. Checkpoint đã đủ epoch sẽ chuyển thẳng sang đánh giá, tránh lỗi `nothing to resume`.
- Khi phiên mới không còn `best.pt` local, notebook tải `best.pt` đã xác minh từ Google Drive; chỉ dùng epoch/last checkpoint làm fallback.

## Tránh lỗi `raytune` trên Kaggle

- Không thêm `raytune=False` vào `model.train(...)`; đây không phải tham số train.
- Không ghi `{"raytune": False}` vào file JSON (`False` không phải cú pháp JSON).
- Notebook cập nhật `settings` theo schema của đúng phiên bản Ultralytics đang chạy. Bản có khóa `raytune` sẽ đặt nó thành `False`; bản cũ thiếu khóa sẽ báo `safe skip` và tiếp tục.
- Pipeline chỉ gọi `model.train()`, không gọi `model.tune(use_ray=True)`, nên không cần cài package `ray[tune]`.

## Train hai giai đoạn

1. Base detector train bằng ba nguồn có bounding box.
2. Đánh giá base trên test group-disjoint.
3. Base teacher pseudo-label 8 ảnh Seatbelt Real; AUC phone-candidate chỉ bật khi có quyền truy cập/mật khẩu hợp lệ, với `conf >= 0.72`.
4. Fine-tune ngắn, giữ nguyên gold validation/test.
5. Chọn checkpoint có `min(mAP50, macro-F1)` cao hơn giữa base và fine-tune.

Nếu GPU hết giờ, giảm theo thứ tự: `IMG_SIZE 640`, `EPOCHS 60`, `MODEL_NAME yolo11s.pt`. Không giảm test integrity hoặc trộn test vào train.

## Nguồn dữ liệu

- Roboflow Primary: <https://universe.roboflow.com/ladailoc-yzh0x/phone-detect-svavs>
- Roboflow Seatbelt & Mobile: <https://universe.roboflow.com/aiactive20092009-gmail-com/seat_belt-and-mobile>
- DMS Safety: <https://www.kaggle.com/datasets/habbas11/dms-driver-monitoring-system>
- AUC Distracted Driver: <https://www.kaggle.com/datasets/tejakalepalle/auc-distracted-driver-dataset-v1>
- Seatbelt Real: <https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos>

## Bảo mật

API key Roboflow cũ từng được lưu trực tiếp trong source/notebook đã được thay bằng biến môi trường. Hãy revoke/rotate key cũ trên Roboflow trước khi đưa repository lên GitHub.

AUC v2 không nằm trong bundle upload: license của MI-AUC cấm chuyển giao/phân phối nếu chưa được cho phép. Hãy xin quyền truy cập từ tác giả và lưu password trong Kaggle Secrets với tên `AUC_ZIP_PASSWORD`; không ghi password vào notebook.
