# Driver Behavior Hybrid — YOLO11 + MediaPipe

Hệ thống giám sát hành vi tài xế, tập trung phát hiện sử dụng điện thoại và không thắt dây an toàn. Pipeline kết hợp YOLO11, MediaPipe Pose, temporal smoothing, backend FastAPI và giao diện React/Vite.

## Cấu trúc

- `backend/app/`: pipeline edge, cloud API và đánh giá.
- `backend/scripts/`: chuẩn hóa dữ liệu, train, pseudo-label và đóng gói Kaggle.
- `backend/driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb`: notebook train Kaggle/Colab.
- `frontend/`: giao diện giám sát và quản lý cảnh báo.
- `backend/docs/kaggle_dms_training_guide.md`: hướng dẫn train trên Kaggle Web.

Dataset, output huấn luyện, checkpoint và credential không được đưa vào Git. Xem `.gitignore` và chuẩn bị chúng riêng trong Kaggle/Google Drive.

## Chạy frontend

```bash
cd frontend
npm install
npm run dev
```

## Chạy backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.cloud.main_cloud:app --reload
```

Chạy pipeline edge:

```bash
cd backend
python -m app.edge.main_edge --config config.yaml
```

## Train trên Kaggle

Làm theo [hướng dẫn Kaggle](backend/docs/kaggle_dms_training_guide.md). Notebook lưu checkpoint từng epoch vào Google Drive thật và tự resume. Chỉ kết luận đạt mục tiêu khi test set độc lập có `mAP@50 >= 0.85` và macro `F1 >= 0.85`.

## Bảo mật và giấy phép dữ liệu

- Không commit `.env`, Kaggle token, Google OAuth JSON, rclone config hoặc API key.
- AUC Distracted Driver không nằm trong bundle Git/Kaggle vì điều kiện phân phối; người chạy phải tự có quyền truy cập hợp lệ.
- Giấy phép của từng nguồn dữ liệu vẫn được áp dụng độc lập với mã nguồn của dự án.
