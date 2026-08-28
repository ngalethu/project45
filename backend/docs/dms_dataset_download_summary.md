# DMS dataset download and harmonization summary

## Kết quả kiểm kê local

| Nguồn | Ảnh có sẵn | Annotation | Cách sử dụng |
|---|---:|---|---|
| Roboflow Primary v9 | 28.282 hợp lệ | YOLO boxes, 4 lớp | Base detector |
| Roboflow Seatbelt & Mobile | 803 | YOLO boxes, 3 lớp | Map mobile/seatbelt; bỏ windshield |
| DMS Safety | 9.884 | YOLO boxes, 5 lớp | Map cigarette/phone/seatbelt; bỏ eye classes |
| AUC Distracted Driver v2 | 32.714 entry, ZIP encrypted | Image classification | Chỉ dùng 5.418 phone-candidate khi có password/quyền hợp lệ |
| Seatbelt Real | 8 | Không có box | Hard-case/pseudo-label |

Detection v2 có 38.969 cặp image/label và 41.938 object instances:

- phone: 14.078
- seatbelt: 10.264
- no-seatbelt: 5.119
- smoking: 12.477

Các con số trên lấy trực tiếp từ file local, không dùng số ước lượng trên mô tả dataset.

## Mapping canonical

Thứ tự class cố định cho train và backend:

```text
0 phone
1 seatbelt
2 no-seatbelt
3 smoking
```

Chi tiết machine-readable nằm trong `data/processed/dms_yolo_4class_v2/audit_report.json` và `dms_dataset.yaml`.
