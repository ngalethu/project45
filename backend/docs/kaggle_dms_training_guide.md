# Hướng dẫn train DMS 3 lớp trên Kaggle T4

## Trạng thái dữ liệu

- Dataset: `data/processed/dms_yolo_3class_v4_12k`
- Train: 12.000 ảnh từ 12.000 capture group duy nhất
- Validation: 2.801 ảnh
- Test: 2.681 ảnh
- Lớp: `phone`, `seatbelt`, `no-seatbelt`
- Negative train: 800 ảnh
- Group giao nhau giữa train/val/test: 0
- Windshield/roadside: giữ toàn bộ 649 group train và 69 ảnh test ngoại miền

Không được kết luận mAP/F1 trên 85% trước khi Kaggle tạo `metrics_summary.json` có `target_met: true`.

## File Kaggle

Upload toàn bộ nội dung thư mục:

`backend/outputs/kaggle_dms_3class_v4_12k_bundle`

Các file chính:

- `dms_yolo_3class_v4_12k.zip`
- `yolo11m.pt`
- `training_code.zip`
- `kaggle_train_dms_3class_12k.ipynb`
- `audit_report.json`

## Cách chạy

1. Tạo Kaggle Dataset mới và upload toàn bộ bundle.
2. Tạo Kaggle Notebook, chọn GPU T4 và attach dataset vừa tạo.
3. Import `kaggle_train_dms_3class_12k.ipynb`.
4. Chạy `Run All`.
5. Notebook mặc định train YOLO11m ở 768 px, batch 8, tối đa 60 epoch, patience 12.
6. Thời gian dự kiến khoảng 4–8 giờ tùy T4 và tốc độ I/O.
7. Tải thư mục `/kaggle/working/dms_export` sau khi hoàn thành.

Kết quả bắt buộc:

- `best.pt`
- `metrics_summary.json`
- `best.onnx` nếu bước export ONNX thành công

## Thay model triển khai

Không chép đè thủ công. Chạy installer để kiểm class và metrics:

```powershell
cd D:\.idea\project4\backend
py -3 scripts\install_kaggle_dms_model.py `
  --best-pt "C:\path\best.pt" `
  --metrics "C:\path\metrics_summary.json" `
  --best-onnx "C:\path\best.onnx"
```

Installer sẽ từ chối checkpoint không đúng ba lớp hoặc chưa đạt target, đồng thời backup model cũ trước khi thay.

## Quy tắc đánh giá

- Metrics ảnh lấy trên test split, không lấy từ train/validation.
- Benchmark video tính một quyết định cho mỗi video/event; không nhân kết quả với số frame.
- Temporal voting dùng cửa sổ 12 frame: phone cần 5 vote, no-seatbelt cần 8 vote.
- Driver-disjoint chưa thể chứng minh vì nguồn công khai không cung cấp driver ID đầy đủ. Hiện chỉ đảm bảo capture-group disjoint; dữ liệu thu mới phải có `camera_id`, `driver_id`, `capture_session_id`.

Chi tiết kỹ thuật và quy trình triển khai nằm trong `backend/docs/dms_12k_kaggle_and_deployment.md`.
