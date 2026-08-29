"""Upgrade the legacy Colab notebook into a Kaggle/Colab multisource workflow."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "driver_behavior_yolo11m_mediapipe_minimal_stable.ipynb"
LEGACY_NOTEBOOK = Path(__file__).resolve().parents[1] / "driver_behavior_yolo11m_mediapipe_minimal_stable_legacy_colab.ipynb"


def cell(cell_type: str, source: str) -> dict:
    payload = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


def markdown(source: str) -> dict:
    return cell("markdown", source)


def code(source: str) -> dict:
    return cell("code", source)


def main() -> None:
    source_notebook = LEGACY_NOTEBOOK if LEGACY_NOTEBOOK.exists() else NOTEBOOK
    notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
    old_cells = notebook["cells"]
    if len(old_cells) < 21:
        raise RuntimeError("Unexpected legacy notebook topology")
    tail_marker = "Thiết kế thuật toán hành vi với MediaPipe"
    tail_start = next(
        (index for index, item in enumerate(old_cells) if tail_marker in "".join(item.get("source", []))),
        14,
    )
    inference_tail = old_cells[tail_start:]

    front = [
        markdown(
            r"""
# Driver Behavior Detection — YOLO11m + MediaPipe (Kaggle/Colab)

Notebook phát triển từ bản `minimal_stable`, giữ nguyên engine YOLO + MediaPipe và nâng cấp phần dữ liệu/huấn luyện cho **4 lớp canonical**:

1. `phone`
2. `seatbelt`
3. `no-seatbelt`
4. `smoking`

## Pipeline mới

- Detection có bounding box: Roboflow Primary v9 + Roboflow Seatbelt/Mobile + DMS Safety.
- Weak source: Seatbelt Real và AUC Distracted Driver (chỉ khi có quyền truy cập/password hợp lệ) được dùng qua pseudo-label có ngưỡng cao.
- Chia lại theo nhóm ảnh/video để tránh leakage giữa train/val/test.
- Train base → đánh giá test → pseudo-label → fine-tune → tự chọn checkpoint tốt hơn.
- Chỉ công bố đạt yêu cầu khi **test mAP@50 ≥ 0.85 và macro F1 ≥ 0.85**.

> `best.pt` dùng đánh giá/suy luận; `last.pt` dùng resume. Không ghi API key trực tiếp vào notebook.
"""
        ),
        code(
            r"""
# 1) MÔI TRƯỜNG TỐI GIẢN — không force-reinstall torch/CUDA
import importlib.util
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

def sh(command: str):
    print(f"\n>>> {command}")
    subprocess.check_call(command, shell=True)

def version_tuple(value: str):
    numbers = []
    for part in value.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits or 0))
    return tuple((numbers + [0, 0, 0])[:3])

installed_ultralytics = None
try:
    installed_ultralytics = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    pass

if (
    installed_ultralytics is None
    or version_tuple(installed_ultralytics) < (8, 3, 0)
    or version_tuple(installed_ultralytics) >= (9, 0, 0)
):
    sh('python -m pip install -q "ultralytics>=8.3,<9" "PyYAML>=6"')
if importlib.util.find_spec("googleapiclient") is None:
    sh('python -m pip install -q "google-api-python-client>=2.100" "google-auth>=2.20"')

# Chỉ cài rclone trên Kaggle khi người dùng thật sự đã gắn secret fallback.
if Path("/kaggle/input").exists() and shutil.which("rclone") is None:
    try:
        from kaggle_secrets import UserSecretsClient
        has_rclone_secret = bool(UserSecretsClient().get_secret("RCLONE_CONFIG_B64"))
    except Exception:
        has_rclone_secret = False
    if has_rclone_secret:
        sh("apt-get update -qq && apt-get install -y -qq rclone")

import cv2
import numpy as np
import torch
import ultralytics

print("Python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("opencv:", cv2.__version__)
print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_CAPABILITY = torch.cuda.get_device_capability(0)
    TORCH_CUDA_ARCHES = torch.cuda.get_arch_list()
    print("GPU:", GPU_NAME)
    print("GPU capability:", GPU_CAPABILITY)
    print("PyTorch CUDA arches:", TORCH_CUDA_ARCHES)
    print("GPU memory (GiB):", round(gpu_props.total_memory / 1024**3, 2))
    if GPU_CAPABILITY >= (12, 0) and "sm_120" not in TORCH_CUDA_ARCHES:
        raise RuntimeError(
            "RTX 50xx/Blackwell cần PyTorch CUDA 12.8+. PyTorch hiện tại thiếu sm_120. "
            "Trên Windows RTX 5090 hãy chạy trong terminal: "
            "python -m pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu130 ; "
            "sau đó Restart Kernel và chạy lại từ đầu."
        )
    # is_available() có thể vẫn True với wheel sai kiến trúc; phép tính thật mới xác minh kernel.
    cuda_probe = torch.randn((256, 256), device="cuda")
    cuda_probe_result = float((cuda_probe @ cuda_probe).mean().item())
    del cuda_probe
    print("CUDA matmul probe: OK", round(cuda_probe_result, 6))
else:
    raise RuntimeError("Hãy bật GPU: Kaggle Settings > Accelerator > GPU")
"""
        ),
        code(
            r"""
# 2) NHẬN DIỆN KAGGLE / COLAB / LOCAL
from pathlib import Path

IS_KAGGLE = Path("/kaggle/input").exists()
IS_COLAB = False
IS_RTX_5090 = "RTX 5090" in GPU_NAME.upper()
if not IS_KAGGLE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        IS_COLAB = True
    except ImportError:
        pass

if IS_KAGGLE:
    PLATFORM = "kaggle"
    INPUT_ROOT = Path("/kaggle/input")
    WORK_ROOT = Path("/kaggle/working/dms_work")
    RUNS_ROOT = Path("/kaggle/working/driver_behavior_runs")
elif IS_COLAB:
    PLATFORM = "colab"
    INPUT_ROOT = Path("/content/drive/MyDrive")
    WORK_ROOT = Path("/content/dms_work")
    RUNS_ROOT = Path("/content/drive/MyDrive/driver_behavior_runs")
else:
    PLATFORM = "local"
    configured_project_root = os.getenv("DMS_PROJECT_ROOT")
    if configured_project_root:
        PROJECT_ROOT = Path(configured_project_root).expanduser().resolve()
    else:
        PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "backend" else Path.cwd().resolve()
    INPUT_ROOT = PROJECT_ROOT
    WORK_ROOT = PROJECT_ROOT / "backend" / "outputs" / "notebook_work"
    # Google Drive for desktop trên máy này mount My Drive tại H:. Có thể đổi bằng
    # biến môi trường DMS_LOCAL_DRIVE_RUNS mà không phải sửa notebook.
    configured_runs_root = os.getenv("DMS_RUNS_ROOT")
    LOCAL_DRIVE_RUNS = Path(os.getenv("DMS_LOCAL_DRIVE_RUNS", r"H:\My Drive\project3_runs"))
    if configured_runs_root:
        RUNS_ROOT = Path(configured_runs_root).expanduser().resolve()
        LOCAL_PERSISTENCE = "configured-persistent-root"
    elif LOCAL_DRIVE_RUNS.parent.exists():
        RUNS_ROOT = LOCAL_DRIVE_RUNS
        LOCAL_PERSISTENCE = "google-drive-desktop"
    else:
        RUNS_ROOT = PROJECT_ROOT / "backend" / "outputs" / "runs_dms"
        LOCAL_PERSISTENCE = "local-fallback"

WORK_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
if IS_KAGGLE:
    working_free_gib = shutil.disk_usage("/kaggle/working").free / 1024**3
    print("Kaggle /working free (GiB):", round(working_free_gib, 2))
    if working_free_gib < 8:
        raise RuntimeError(
            "Kaggle /working còn dưới 8 GiB; hãy Restart Session trước khi giải nén/train."
        )
print("PLATFORM =", PLATFORM)
print("INPUT_ROOT =", INPUT_ROOT)
print("WORK_ROOT =", WORK_ROOT)
print("RUNS_ROOT =", RUNS_ROOT)
print("RTX_5090_PROFILE =", IS_RTX_5090)
if PLATFORM == "local":
    print("LOCAL_PERSISTENCE =", LOCAL_PERSISTENCE)
"""
        ),
        markdown(
            r"""
## Dữ liệu multisource

Trên Kaggle, hãy attach dataset bundle chứa:

- `dms_yolo_4class_v2.zip`: 38.969 ảnh detection hợp lệ đã harmonize.
- `seatbelt_real_unlabelled.zip`: 8 hard-case chưa có bounding box.
- `training_code.zip`: script audit/train/pseudo-label.
- `yolo11m.pt`: pretrained checkpoint để chạy được cả khi notebook tắt Internet.

Notebook không tự gán full-image box cho AUC vì đó là nhãn classification, không phải object detection.
AUC không được đóng gói lại vì license cấm phân phối; attach riêng nguồn được cấp quyền và lưu password trong Kaggle Secret `AUC_ZIP_PASSWORD`.
"""
        ),
        code(
            r"""
# 3) CẤU HÌNH CHÍNH
DATASET_ARCHIVE_NAME = "dms_yolo_4class_v2.zip"
AUC_ARCHIVE_NAME = "auc.distracted.driver.dataset_v2.zip"
SEATBELT_ARCHIVE_NAME = "seatbelt_real_unlabelled.zip"
TRAINING_CODE_ARCHIVE_NAME = "training_code.zip"

EXP_NAME = "yolo11m_dms_4class_base"
FINE_TUNE_NAME = "yolo11m_dms_4class_pseudo_finetune"
MODEL_NAME = "yolo11m.pt"
RESUME_CKPT = None
FINE_RESUME_CKPT = None

# Google Drive persistence. Folder ID lấy từ link người dùng cung cấp.
DRIVE_FOLDER_ID = "1RfDV984zjw0Y5yfnxtnd7pPQhJpNczt_"
DRIVE_OAUTH_SECRET = "GDRIVE_OAUTH_JSON"
DRIVE_SERVICE_ACCOUNT_SECRET = "GDRIVE_SERVICE_ACCOUNT_JSON"
RCLONE_CONFIG_B64_SECRET = "RCLONE_CONFIG_B64"
# Dùng remote name thuần để tương thích cả rclone cũ trong Kaggle.
# Folder ID được truyền qua biến backend chính thức RCLONE_DRIVE_ROOT_FOLDER_ID;
# tránh connection-string mới ``gdrive,root_folder_id=...:`` mà rclone cũ
# hiểu nhầm thành một config name chứa dấu phẩy.
RCLONE_REMOTE = os.getenv("DMS_RCLONE_REMOTE", "gdrive:").rstrip("/")
RCLONE_DRIVE_ROOT_FOLDER_ID = os.getenv("DMS_RCLONE_ROOT_FOLDER_ID", DRIVE_FOLDER_ID)
DRIVE_SYNC_ENABLED = True
DRIVE_SYNC_REQUIRED = IS_KAGGLE or (
    IS_RTX_5090 and os.getenv("DMS_REQUIRE_REMOTE_CHECKPOINT", "1") != "0"
)
RCLONE_FALLBACK_ENABLED = True
AUTO_RESUME_FROM_STORE = True

# Giới hạn toàn pipeline trong gói máy 5–6 giờ. Ultralytics `time` tính theo giờ
# và override `epochs`; nếu đủ epoch sớm thì dừng sớm theo epoch bình thường.
IMG_SIZE = 640
EPOCHS = 80
PSEUDO_EPOCHS = 20
TOTAL_BUDGET_HOURS = float(os.getenv("DMS_TOTAL_BUDGET_HOURS", "5.5"))
BASE_TRAIN_HOURS = float(os.getenv("DMS_BASE_TRAIN_HOURS", "4.5"))
FINE_TRAIN_HOURS = float(os.getenv("DMS_FINE_TRAIN_HOURS", "0.5"))
if BASE_TRAIN_HOURS + FINE_TRAIN_HOURS > TOTAL_BUDGET_HOURS - 0.25:
    raise ValueError("Phải dành ít nhất 0.25 giờ cho test/export/upload artifacts")
BATCH = 32 if IS_RTX_5090 else 8
WORKERS = 4 if IS_RTX_5090 else 2
PATIENCE = 20
CACHE = False
SAVE_PERIOD = 1      # bắt buộc tạo checkpoint sau từng epoch
SEED = 42

PSEUDO_LABEL_ENABLED = True
PSEUDO_CONF = 0.72
PSEUDO_LIMIT = None  # None = dùng toàn bộ 5.418 AUC phone-candidate + 8 Seatbelt Real

DET_CONF = 0.35
MP_MODEL_COMPLEXITY = 1
MP_MIN_DET_CONF = 0.45
MP_MIN_TRACK_CONF = 0.45
MP_MIN_VIS = 0.35
POSE_BRIGHTEN_GAMMA = 1.20
USE_BRIGHTEN_FOR_POSE = True

print("Base:", MODEL_NAME, IMG_SIZE, EPOCHS)
print("Pseudo:", PSEUDO_LABEL_ENABLED, PSEUDO_CONF, PSEUDO_EPOCHS)
print("Runtime profile:", GPU_NAME, "batch=", BATCH, "workers=", WORKERS)
print("Time budget (hours): total=", TOTAL_BUDGET_HOURS, "base=", BASE_TRAIN_HOURS, "fine=", FINE_TRAIN_HOURS)
"""
        ),
        markdown(
            r"""
## Lưu checkpoint từng epoch vào Google Drive thật

Folder đích: [DMS checkpoints](https://drive.google.com/drive/folders/1RfDV984zjw0Y5yfnxtnd7pPQhJpNczt_?usp=sharing).

- **Máy Windows này:** notebook ghi trực tiếp vào `H:\My Drive\project3_runs` (Google Drive for desktop). Có thể đổi bằng biến môi trường `DMS_LOCAL_DRIVE_RUNS`.
- **Kaggle:** ưu tiên Google Drive API. Nếu không có OAuth JSON, notebook có thể dùng `rclone` đã cài và secret `RCLONE_CONFIG_B64`.
- `/kaggle/working` chỉ là scratch disk; notebook dừng trước khi train nếu không xác minh được một backend lưu trữ thật.

### Kaggle + My Drive cá nhân (khuyến nghị)

1. Trong Google Cloud Console, bật **Google Drive API** và tạo OAuth Client loại **Desktop app**.
2. Cấp quyền Drive một lần để lấy refresh token của chính tài khoản sở hữu/có quyền Editor trên folder.
3. Trong Kaggle → Add-ons → Secrets, tạo secret `GDRIVE_OAUTH_JSON` và attach cho notebook:

```json
{
  "client_id": "...apps.googleusercontent.com",
  "client_secret": "...",
  "refresh_token": "1//...",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

Không in secret vào cell/output và không lưu JSON credential trong Kaggle Dataset.
Nếu OAuth consent screen còn ở trạng thái **Testing**, refresh token cho scope Drive sẽ hết hạn sau 7 ngày; để resume lâu dài, dùng ứng dụng Internal của Google Workspace hoặc chuyển publishing status phù hợp cho ứng dụng cá nhân.

### rclone dự phòng trên Kaggle

Trên máy local đã cấu hình remote `gdrive:`. Để dùng cùng remote trên Kaggle, mã hóa nội dung `rclone.conf` thành base64 và lưu trong Kaggle Secret `RCLONE_CONFIG_B64`; không upload file cấu hình chứa token vào Dataset. Notebook giải mã secret vào file tạm quyền `0600`, gọi `rclone` bằng danh sách đối số (không qua shell), và kiểm tra size/MD5 sau upload.

Nếu dùng rclone, binary `rclone` phải có sẵn trong image hoặc được attach/cài trước cell checkpoint. Google Drive API vẫn là đường chính vì không cần mang cấu hình rclone lên Kaggle.

### Service account

Chỉ dùng `GDRIVE_SERVICE_ACCOUNT_JSON` khi folder nằm trong **Shared Drive** và service account đã được cấp quyền Contributor/Content manager. Google service account không có quota sở hữu file trong My Drive cá nhân, vì vậy notebook chủ động từ chối cấu hình đó thay vì train rồi mất checkpoint.

Trên Kaggle phải bật Internet để Drive API hoạt động. Cell kế tiếp thực hiện write probe và đọc lại metadata; nếu thất bại, quá trình dừng **trước khi train**.
"""
        ),
        code(
            r"""
# 4) GOOGLE DRIVE CHECKPOINT STORE — API thật, không giả lập thư mục Drive trên Kaggle
import base64
import hashlib
import io
import json
import mimetypes
import re
import shutil
import uuid
from datetime import datetime, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

DRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]

def notebook_secret(name: str):
    # Đọc secret mà không ghi credential ra file hoặc log.
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            return None
    if IS_COLAB:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            return None
    return os.getenv(name)

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def drive_query_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")

class DriveRunStore:
    def __init__(self, parent, folder_id: str, run_name: str):
        self.parent = parent
        self.drive = parent.drive
        self.folder_id = folder_id
        self.run_name = run_name

    def _find(self, name: str):
        q = (
            f"'{self.folder_id}' in parents and trashed = false and "
            f"name = '{drive_query_literal(name)}'"
        )
        response = self.drive.files().list(
            q=q, spaces="drive", pageSize=100,
            fields="files(id,name,size,md5Checksum,webViewLink)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute(num_retries=5)
        return response.get("files", [])

    def upload(self, local_path, remote_name=None):
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Không thấy file cần upload: {local_path}")
        remote_name = remote_name or local_path.name
        local_size = local_path.stat().st_size
        local_md5 = md5_file(local_path)
        matches = self._find(remote_name)
        if len(matches) > 1:
            raise RuntimeError(f"Drive có nhiều file trùng tên {remote_name} trong run {self.run_name}")
        if matches and matches[0].get("md5Checksum") == local_md5:
            return matches[0]

        mime_type = mimetypes.guess_type(remote_name)[0] or "application/octet-stream"
        media = MediaFileUpload(
            str(local_path), mimetype=mime_type,
            chunksize=8 * 1024 * 1024, resumable=True,
        )
        if matches:
            request = self.drive.files().update(
                fileId=matches[0]["id"], media_body=media,
                fields="id,name,size,md5Checksum,webViewLink",
                supportsAllDrives=True,
            )
        else:
            request = self.drive.files().create(
                body={"name": remote_name, "parents": [self.folder_id]},
                media_body=media,
                fields="id,name,size,md5Checksum,webViewLink",
                supportsAllDrives=True,
            )
        uploaded = request.execute(num_retries=5)
        verified = self.drive.files().get(
            fileId=uploaded["id"], fields="id,name,size,md5Checksum,webViewLink,parents",
            supportsAllDrives=True,
        ).execute(num_retries=5)
        if int(verified.get("size", -1)) != local_size:
            raise IOError(f"Drive size mismatch cho {remote_name}")
        if verified.get("md5Checksum") and verified["md5Checksum"] != local_md5:
            raise IOError(f"Drive MD5 mismatch cho {remote_name}")
        print(f"[DRIVE OK] {self.run_name}/{remote_name} ({local_size / 1e6:.1f} MB)")
        return verified

    def list_checkpoints(self):
        q = f"'{self.folder_id}' in parents and trashed = false"
        response = self.drive.files().list(
            q=q, spaces="drive", pageSize=1000,
            fields="files(id,name,size,md5Checksum,modifiedTime)",
            orderBy="modifiedTime desc",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute(num_retries=5)
        return response.get("files", [])

    def download_named(self, remote_name, destination_dir, required=False):
        matches = self._find(remote_name)
        if not matches:
            if required:
                raise FileNotFoundError(f"Drive không có {self.run_name}/{remote_name}")
            return None
        if len(matches) > 1:
            raise RuntimeError(f"Drive có nhiều file trùng tên {remote_name} trong run {self.run_name}")
        item = matches[0]
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / remote_name
        request = self.drive.files().get_media(fileId=item["id"], supportsAllDrives=True)
        with destination.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=5)
        if int(item.get("size", -1)) != destination.stat().st_size:
            raise IOError(f"File tải từ Drive sai kích thước: {destination}")
        if item.get("md5Checksum") and item["md5Checksum"] != md5_file(destination):
            raise IOError(f"File tải từ Drive sai MD5: {destination}")
        print(f"[DRIVE DOWNLOAD] {self.run_name}/{remote_name}: {destination}")
        return destination

    def download_latest(self, destination_dir):
        pattern = re.compile(r"epoch_(\d+)\.pt$")
        candidates = []
        for item in self.list_checkpoints():
            match = pattern.fullmatch(item.get("name", ""))
            if match:
                candidates.append((int(match.group(1)), item))
        if not candidates:
            print(f"[DRIVE] Chưa có checkpoint cũ cho {self.run_name}")
            return None
        epoch, item = max(candidates, key=lambda pair: pair[0])
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / item["name"]
        request = self.drive.files().get_media(
            fileId=item["id"], supportsAllDrives=True,
        )
        with destination.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=5)
        if int(item.get("size", -1)) != destination.stat().st_size:
            raise IOError(f"Checkpoint tải từ Drive sai kích thước: {destination}")
        if item.get("md5Checksum") and item["md5Checksum"] != md5_file(destination):
            raise IOError(f"Checkpoint tải từ Drive sai MD5: {destination}")
        print(f"[DRIVE RESUME] epoch {epoch}: {destination}")
        return destination

class DriveCheckpointStore:
    def __init__(self, folder_id: str, oauth_json=None, service_account_json=None):
        if oauth_json:
            info = json.loads(oauth_json)
            credentials = UserCredentials(
                token=None,
                refresh_token=info["refresh_token"],
                token_uri=info.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=info["client_id"],
                client_secret=info["client_secret"],
                scopes=DRIVE_SCOPE,
            )
            credentials.refresh(Request())
            self.auth_mode = "oauth-user"
            self.identity = info.get("account", "Google user OAuth")
        elif service_account_json:
            info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPE)
            self.auth_mode = "service-account"
            self.identity = info.get("client_email", "service account")
        else:
            raise RuntimeError("Thiếu Google Drive credential")

        self.drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.root_folder_id = folder_id
        metadata = self.drive.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,driveId,capabilities(canAddChildren)",
            supportsAllDrives=True,
        ).execute(num_retries=5)
        if metadata.get("mimeType") != "application/vnd.google-apps.folder":
            raise ValueError(f"DRIVE_FOLDER_ID không phải folder: {metadata}")
        if not metadata.get("capabilities", {}).get("canAddChildren", False):
            raise PermissionError("Credential không có quyền Editor/canAddChildren trên folder Drive")
        if self.auth_mode == "service-account" and not metadata.get("driveId"):
            raise RuntimeError(
                "Service account không được dùng với My Drive vì không có storage quota. "
                "Hãy dùng GDRIVE_OAUTH_JSON, hoặc chuyển folder sang Shared Drive."
            )
        print(f"[DRIVE VERIFIED] {metadata['name']} | auth={self.auth_mode} | identity={self.identity}")

    def _find_folder(self, name: str):
        q = (
            f"'{self.root_folder_id}' in parents and trashed = false and "
            "mimeType = 'application/vnd.google-apps.folder' and "
            f"name = '{drive_query_literal(name)}'"
        )
        response = self.drive.files().list(
            q=q, spaces="drive", pageSize=10, fields="files(id,name)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute(num_retries=5)
        return response.get("files", [])

    def for_run(self, run_name: str):
        folders = self._find_folder(run_name)
        if len(folders) > 1:
            raise RuntimeError(f"Drive có nhiều thư mục run trùng tên: {run_name}")
        if folders:
            folder_id = folders[0]["id"]
        else:
            created = self.drive.files().create(
                body={
                    "name": run_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [self.root_folder_id],
                },
                fields="id,name", supportsAllDrives=True,
            ).execute(num_retries=5)
            folder_id = created["id"]
        return DriveRunStore(self, folder_id, run_name)

def rclone_call(*arguments: str, capture_output: bool = False):
    # Chạy rclone an toàn, không dùng shell và không in credential.
    return subprocess.run(
        ["rclone", *map(str, arguments)],
        check=True,
        text=True,
        capture_output=capture_output,
    )

def rclone_mkdir(remote: str):
    subprocess.run(
        [
            "rclone",
            "mkdir",
            remote,
        ],
        check=True,
    )

def configure_rclone_secret():
    # Áp dụng folder ID cả khi dùng rclone.conf có sẵn trên máy RTX 5090.
    os.environ["RCLONE_DRIVE_ROOT_FOLDER_ID"] = RCLONE_DRIVE_ROOT_FOLDER_ID
    encoded = notebook_secret(RCLONE_CONFIG_B64_SECRET)
    if not encoded:
        return None
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"Secret {RCLONE_CONFIG_B64_SECRET} không phải base64 hợp lệ") from exc
    secret_dir = WORK_ROOT / ".secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    config_path = secret_dir / "rclone.conf"
    config_path.write_bytes(payload)
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    os.environ["RCLONE_CONFIG"] = str(config_path)
    # Backend option documented by rclone; works on older Kaggle packages and
    # restricts gdrive: to the exact folder supplied by the user.
    return config_path

class RcloneRunStore:
    def __init__(self, parent, remote_folder: str, run_name: str):
        self.parent = parent
        self.remote_folder = remote_folder.rstrip("/")
        self.run_name = run_name

    def _target(self, name: str) -> str:
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Tên file remote không hợp lệ: {name!r}")
        return f"{self.remote_folder}/{name}"

    def _stat(self, name: str):
        result = rclone_call("lsjson", self._target(name), "--files-only", "--hash", capture_output=True)
        items = json.loads(result.stdout or "[]")
        if len(items) != 1:
            raise IOError(f"Không xác minh được remote file {self._target(name)}")
        return items[0]

    @staticmethod
    def _md5(item):
        hashes = item.get("Hashes") or {}
        return hashes.get("md5") or hashes.get("MD5")

    def upload(self, local_path, remote_name=None):
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Không thấy file cần upload: {local_path}")
        remote_name = remote_name or local_path.name
        local_size = local_path.stat().st_size
        local_md5 = md5_file(local_path)
        try:
            current = self._stat(remote_name)
        except (subprocess.CalledProcessError, IOError):
            current = None
        if current:
            current_md5 = self._md5(current)
            if int(current.get("Size", -1)) == local_size and current_md5 and current_md5.lower() == local_md5.lower():
                return {
                    "id": self._target(remote_name), "name": remote_name,
                    "size": str(local_size), "md5Checksum": current_md5,
                }
        final_target = self._target(remote_name)
        temporary_name = f".{remote_name}.{uuid.uuid4().hex}.uploading"
        temporary_target = self._target(temporary_name)
        try:
            rclone_call(
                "copyto", str(local_path), temporary_target,
                "--checksum", "--retries", "5", "--low-level-retries", "10",
            )
            temporary = self._stat(temporary_name)
            if int(temporary.get("Size", -1)) != local_size:
                raise IOError(f"rclone size mismatch cho {remote_name}")
            remote_md5 = self._md5(temporary)
            if remote_md5 and remote_md5.lower() != local_md5.lower():
                raise IOError(f"rclone MD5 mismatch cho {remote_name}")
            rclone_call("moveto", temporary_target, final_target, "--retries", "5")
        finally:
            # Xóa duy nhất file staging của lần upload này nếu nó còn tồn tại.
            subprocess.run(
                ["rclone", "deletefile", temporary_target],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        verified = self._stat(remote_name)
        if int(verified.get("Size", -1)) != local_size:
            raise IOError(f"rclone final size mismatch cho {remote_name}")
        final_md5 = self._md5(verified)
        if final_md5 and final_md5.lower() != local_md5.lower():
            raise IOError(f"rclone final MD5 mismatch cho {remote_name}")
        print(f"[RCLONE OK] {self.run_name}/{remote_name} ({local_size / 1e6:.1f} MB)")
        return {
            "id": final_target,
            "name": remote_name,
            "size": str(verified.get("Size", local_size)),
            "md5Checksum": final_md5,
        }

    def list_checkpoints(self):
        result = rclone_call("lsjson", self.remote_folder, "--files-only", "--hash", capture_output=True)
        return json.loads(result.stdout or "[]")

    def download_named(self, remote_name, destination_dir, required=False):
        try:
            item = self._stat(remote_name)
        except (subprocess.CalledProcessError, IOError):
            if required:
                raise FileNotFoundError(f"rclone không có {self.run_name}/{remote_name}")
            return None
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / remote_name
        rclone_call("copyto", self._target(remote_name), str(destination), "--retries", "5")
        if int(item.get("Size", -1)) != destination.stat().st_size:
            raise IOError(f"File tải bằng rclone sai kích thước: {destination}")
        remote_md5 = self._md5(item)
        if remote_md5 and remote_md5.lower() != md5_file(destination).lower():
            raise IOError(f"File tải bằng rclone sai MD5: {destination}")
        print(f"[RCLONE DOWNLOAD] {self.run_name}/{remote_name}: {destination}")
        return destination

    def download_latest(self, destination_dir):
        pattern = re.compile(r"epoch_(\d+)\.pt$")
        candidates = []
        for item in self.list_checkpoints():
            match = pattern.fullmatch(item.get("Name", ""))
            if match:
                candidates.append((int(match.group(1)), item))
        if not candidates:
            print(f"[RCLONE] Chưa có checkpoint cũ cho {self.run_name}")
            return None
        epoch, item = max(candidates, key=lambda pair: pair[0])
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / item["Name"]
        rclone_call("copyto", self._target(item["Name"]), str(destination), "--retries", "5")
        if int(item.get("Size", -1)) != destination.stat().st_size:
            raise IOError(f"Checkpoint tải bằng rclone sai kích thước: {destination}")
        remote_md5 = self._md5(item)
        if remote_md5 and remote_md5.lower() != md5_file(destination).lower():
            raise IOError(f"Checkpoint tải bằng rclone sai MD5: {destination}")
        print(f"[RCLONE RESUME] epoch {epoch}: {destination}")
        return destination

class RcloneCheckpointStore:
    def __init__(self, remote_root: str):
        if shutil.which("rclone") is None:
            raise RuntimeError("Không tìm thấy binary rclone")
        if ":" not in remote_root:
            raise ValueError(f"RCLONE_REMOTE phải có dạng remote:path, nhận được {remote_root!r}")
        self.remote_root = remote_root.rstrip("/")
        rclone_mkdir(self.remote_root)
        rclone_call("lsjson", self.remote_root, "--max-depth", "1", capture_output=True)
        print(f"[RCLONE VERIFIED] {self.remote_root}")

    def for_run(self, run_name: str):
        if not run_name or "/" in run_name or "\\" in run_name:
            raise ValueError(f"Tên run không hợp lệ: {run_name!r}")
        remote_folder = f"{self.remote_root}/{run_name}"
        listing = rclone_call(
            "lsjson", self.remote_root, "--dirs-only", "--max-depth", "1",
            capture_output=True,
        )
        matches = [item for item in json.loads(listing.stdout or "[]") if item.get("Name") == run_name]
        if len(matches) > 1:
            raise RuntimeError(f"rclone thấy nhiều thư mục run trùng tên: {run_name}")
        if not matches:
            rclone_mkdir(remote_folder)
        return RcloneRunStore(self, remote_folder, run_name)

def attach_checkpoint_callbacks(model, run_name: str):
    if CHECKPOINT_STORE is None:
        return None
    run_store = CHECKPOINT_STORE.for_run(run_name)
    uploaded_epochs = set()

    def upload_common(trainer):
        for path in (Path(trainer.save_dir) / "args.yaml", Path(trainer.save_dir) / "results.csv"):
            if path.is_file():
                run_store.upload(path)
        best = Path(trainer.best)
        if best.is_file():
            run_store.upload(best, "best.pt")

    def on_fit_epoch_end(trainer):
        epoch_number = int(trainer.epoch) + 1
        if epoch_number in uploaded_epochs:
            return
        epoch_file = Path(trainer.wdir) / f"epoch{trainer.epoch}.pt"
        checkpoint = epoch_file if epoch_file.is_file() else Path(trainer.last)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Epoch {epoch_number} kết thúc nhưng không thấy checkpoint")
        remote = run_store.upload(checkpoint, f"epoch_{epoch_number:03d}.pt")
        upload_common(trainer)
        manifest = Path(trainer.save_dir) / "checkpoint_latest.json"
        manifest.write_text(json.dumps({
            "run": run_name,
            "epoch": epoch_number,
            "checkpoint_name": remote["name"],
            "checkpoint_id": remote["id"],
            "checkpoint_md5": remote.get("md5Checksum"),
            "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        run_store.upload(manifest, "checkpoint_latest.json")
        uploaded_epochs.add(epoch_number)

    def on_train_end(trainer):
        upload_common(trainer)
        last = Path(trainer.last)
        if last.is_file():
            run_store.upload(last, "last.pt")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    model._checkpoint_run_store = run_store
    print(f"[CHECKPOINT] Callback từng epoch đã bật cho {run_name}")
    return run_store

CHECKPOINT_STORE = None
if IS_KAGGLE:
    kaggle_runs = RUNS_ROOT.resolve()
    if not kaggle_runs.is_relative_to(Path("/kaggle/working")):
        raise RuntimeError(f"Kaggle RUNS_ROOT phải nằm trong /kaggle/working: {kaggle_runs}")
    print("[KAGGLE] RUNS_ROOT chỉ là scratch disk; Drive persistence chỉ hợp lệ qua API callback.")

if DRIVE_SYNC_ENABLED:
    oauth_json = notebook_secret(DRIVE_OAUTH_SECRET)
    service_json = notebook_secret(DRIVE_SERVICE_ACCOUNT_SECRET)
    if oauth_json or service_json:
        CHECKPOINT_STORE = DriveCheckpointStore(
            DRIVE_FOLDER_ID, oauth_json=oauth_json, service_account_json=service_json,
        )
    elif RCLONE_FALLBACK_ENABLED:
        configure_rclone_secret()
        if shutil.which("rclone") is not None:
            CHECKPOINT_STORE = RcloneCheckpointStore(RCLONE_REMOTE)

    if CHECKPOINT_STORE is not None:
        probe_store = CHECKPOINT_STORE.for_run(EXP_NAME)
        probe = WORK_ROOT / "checkpoint_sync_ready.json"
        probe.write_text(json.dumps({
            "status": "ready", "platform": PLATFORM,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        probe_store.upload(probe, "checkpoint_sync_ready.json")
        print("[CHECKPOINT READY] Đã kiểm tra ghi và đọc metadata thành công.")
    elif DRIVE_SYNC_REQUIRED:
        raise RuntimeError(
            f"{PLATFORM} chưa có persistence thật. Cấu hình {DRIVE_OAUTH_SECRET}, hoặc "
            f"cài rclone và cung cấp {RCLONE_CONFIG_B64_SECRET}/remote gdrive. Không bắt đầu train."
        )
    else:
        print(f"[DIRECT PERSISTENCE] Checkpoint được ghi trực tiếp tại {RUNS_ROOT}")
"""
        ),
        code(
            r"""
# 5) GIẢI NÉN DATASET/CODE VÀ TÌM PRETRAINED WEIGHTS
import shutil
import sys
import zipfile

def find_input_file(filename: str, required: bool = True):
    candidates = sorted(INPUT_ROOT.rglob(filename)) if INPUT_ROOT.exists() else []
    if candidates:
        print(f"{filename}: {candidates[0]}")
        return candidates[0]
    if required:
        raise FileNotFoundError(f"Không thấy {filename} trong {INPUT_ROOT}. Hãy attach Kaggle dataset bundle.")
    return None

def find_input_dir(dirname: str, required: bool = False):
    candidates = sorted(
        path for path in INPUT_ROOT.rglob(dirname)
        if path.is_dir()
    ) if INPUT_ROOT.exists() else []
    if candidates:
        print(f"{dirname}/: {candidates[0]}")
        return candidates[0]
    if required:
        raise FileNotFoundError(f"Không thấy thư mục {dirname} trong {INPUT_ROOT}.")
    return None

local_v2 = Path(r"D:/.idea/project4/data/processed/dms_yolo_4class_v2")
if PLATFORM == "local" and (local_v2 / "dms_dataset.yaml").exists():
    dataset_dir_used = local_v2
else:
    # Kaggle Upload Data thường tự giải nén ZIP thành một hoặc nhiều thư mục lồng nhau.
    # Ưu tiên YAML đã attach; chỉ giải nén thủ công khi Input thực sự còn file ZIP.
    attached_yamls = sorted(INPUT_ROOT.rglob("dms_dataset.yaml")) if INPUT_ROOT.exists() else []
    if len(attached_yamls) == 1:
        dataset_dir_used = attached_yamls[0].parent
        print("Kaggle extracted dataset:", dataset_dir_used)
    elif len(attached_yamls) > 1:
        raise RuntimeError(f"Có nhiều dms_dataset.yaml trong Kaggle Input: {attached_yamls}")
    else:
        dataset_zip = find_input_file(DATASET_ARCHIVE_NAME)
        extract_root = WORK_ROOT / "dataset"
        yaml_candidates = list(extract_root.rglob("dms_dataset.yaml")) if extract_root.exists() else []
        if not yaml_candidates:
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dataset_zip) as archive:
                archive.extractall(extract_root)
            yaml_candidates = list(extract_root.rglob("dms_dataset.yaml"))
        if len(yaml_candidates) != 1:
            raise RuntimeError(f"Cần đúng 1 dms_dataset.yaml, tìm thấy: {yaml_candidates}")
        dataset_dir_used = yaml_candidates[0].parent

# Kaggle cũng có thể tự giải nén training_code.zip; ưu tiên script đã attach.
attached_training_scripts = sorted(INPUT_ROOT.rglob("train_yolo11_dms.py")) if INPUT_ROOT.exists() else []
if attached_training_scripts:
    training_code_dir = attached_training_scripts[0].parent
    print("Kaggle extracted training code:", training_code_dir)
else:
    training_code_dir = WORK_ROOT / "training_code"
    code_zip = find_input_file(TRAINING_CODE_ARCHIVE_NAME, required=False)
    if code_zip and not list(training_code_dir.rglob("train_yolo11_dms.py")):
        training_code_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(code_zip) as archive:
            archive.extractall(training_code_dir)
if PLATFORM == "local" and (PROJECT_ROOT / "backend" / "scripts" / "train_yolo11_dms.py").exists():
    training_code_dir = PROJECT_ROOT / "backend" / "scripts"
sys.path.insert(0, str(training_code_dir))

attached_model = find_input_file(MODEL_NAME, required=False)
MODEL_SIZE = str(attached_model or MODEL_NAME)
print("dataset_dir_used =", dataset_dir_used)
print("MODEL_SIZE =", MODEL_SIZE)
"""
        ),
        code(
            r"""
# 6) XÁC THỰC ONTOLOGY + TẠO YAML RUNTIME GHI ĐƯỢC
import yaml

source_yaml = Path(dataset_dir_used) / "dms_dataset.yaml"
cfg = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
names_raw = cfg.get("names")
if isinstance(names_raw, dict):
    names = [str(names_raw[key]) for key in sorted(names_raw, key=lambda x: int(x))]
else:
    names = list(names_raw or [])

EXPECTED_NAMES = ["phone", "seatbelt", "no-seatbelt", "smoking"]
assert names == EXPECTED_NAMES, f"Sai ontology/order: {names} != {EXPECTED_NAMES}"
assert int(cfg.get("nc", 0)) == 4, f"nc phải bằng 4, hiện tại: {cfg.get('nc')}"

dataset_root = Path(cfg.get("path") or ".")
if not dataset_root.is_absolute():
    dataset_root = (source_yaml.parent / dataset_root).resolve()
cfg["path"] = dataset_root.as_posix()
for split in ("train", "val", "test"):
    image_dir = dataset_root / cfg[split]
    assert image_dir.exists(), f"Thiếu {split}: {image_dir}"

yaml_path = WORK_ROOT / "runtime_dms_dataset.yaml"
yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(yaml_path.read_text(encoding="utf-8"))
"""
        ),
        markdown(
            r"""
## Label harmonization đã áp dụng trước notebook

| Source | Mapping sang canonical | Nhãn loại bỏ |
|---|---|---|
| Roboflow Primary v9 | `0→no-seatbelt`, `1→phone`, `2→seatbelt`, `3→smoking` | — |
| Seatbelt & Mobile | `mobile→phone`, `seatbelt→seatbelt` | `windshield` |
| DMS Safety | `cigarette→smoking`, `phone→phone`, `seatbelt→seatbelt` | `open-eye`, `closed-eye` |
| AUC / Seatbelt Real | teacher pseudo-label, `conf ≥ 0.72` | ảnh không có box tin cậy |

Không sửa riêng tên trong YAML nếu chưa remap class id trong từng label.
"""
        ),
        code(
            r"""
# 7) KIỂM TRA CẶP IMAGE/LABEL VÀ PHÂN BỐ CLASS
from collections import Counter

class_instances = Counter()
split_images = {}
split_labels = {}
for split in ("train", "val", "test"):
    image_dir = dataset_root / cfg[split]
    label_dir = Path(str(image_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    labels = list(label_dir.glob("*.txt"))
    split_images[split] = len(images)
    split_labels[split] = len(labels)
    assert len(images) == len(labels), f"{split}: images={len(images)} labels={len(labels)}"
    for label in labels:
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                assert 0 <= class_id < 4, f"Class id lỗi {class_id}: {label}"
                class_instances[EXPECTED_NAMES[class_id]] += 1

print("Images:", split_images)
print("Labels:", split_labels)
print("Instances:", dict(class_instances))
"""
        ),
        code(
            r"""
# 8) ĐỌC AUDIT REPORT — leakage cũ phải được xử lý bằng grouped split
import json

audit_path = Path(dataset_dir_used) / "audit_report.json"
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for key in (
        "images_total", "detection_images", "weak_images", "source_images",
        "class_instances", "dropped_instances", "groups_crossing_original_splits", "build"
    ):
        print(f"{key}: {audit.get(key)}")
    assert audit.get("invalid_label_lines", 0) == 0
    assert audit.get("unknown_class_lines", 0) == 0
    assert audit.get("build", {}).get("skipped_unreadable_image", 0) == 0
else:
    print("Không có audit_report.json; cell 6 vẫn đã kiểm tra pairing/class id.")
"""
        ),
        markdown(
            r"""
## Train / Resume

- Kaggle: `/kaggle/working` chỉ là scratch disk. Mỗi epoch phải được callback xác minh upload qua Google Drive API hoặc rclone.
- Colab: checkpoint nằm trong Google Drive.
- Local: checkpoint nằm trực tiếp tại `H:\My Drive\project3_runs` nếu ổ Drive Desktop đang mount.
- Resume bằng `last.pt`; đánh giá/suy luận bằng `best.pt`.
- Khi `AUTO_RESUME_FROM_STORE=True`, notebook tải checkpoint `epoch_NNN.pt` mới nhất đã xác minh MD5; với local/Colab, ưu tiên `weights/last.pt` tại thư mục đồng bộ trực tiếp.
- Nếu checkpoint đã đủ số epoch yêu cầu, notebook không gọi `resume=True` lần nữa mà tải `best.pt` và chuyển thẳng sang đánh giá.
- Cả base stage và pseudo fine-tune stage đều tự resume độc lập.
- Test split chỉ dùng đánh giá cuối, không dùng chọn pseudo-label hoặc tune threshold.
"""
        ),
        code(
            r"""
# 9) TRAIN BASE DETECTOR
import torch
from ultralytics import YOLO, settings

def update_ultralytics_settings_compat(**requested):
    # Một số Kaggle image giữ Ultralytics cũ, nơi khóa `raytune` chưa tồn tại.
    # SettingsManager sẽ raise nếu update khóa lạ, nên chỉ gửi các khóa được hỗ trợ.
    available = set(settings.keys())
    supported = {key: value for key, value in requested.items() if key in available}
    skipped = sorted(set(requested) - set(supported))
    if supported:
        settings.update(supported)
    print("Ultralytics integrations disabled:", sorted(supported))
    if skipped:
        print("Ultralytics settings not present in this version (safe skip):", skipped)
    return supported

ULTRALYTICS_DISABLED = update_ultralytics_settings_compat(
    wandb=False,
    raytune=False,
    sync=False,
)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
save_dir = RUNS_ROOT / EXP_NAME

def checkpoint_epoch_number(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    match = re.fullmatch(r"epoch_(\d+)\.pt", checkpoint_path.name)
    if match:
        return int(match.group(1))
    try:
        try:
            payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            # PyTorch < 2.0 does not expose the weights_only keyword.
            payload = torch.load(str(checkpoint_path), map_location="cpu")
        return int(payload.get("epoch", -1)) + 1 if isinstance(payload, dict) else 0
    except Exception as exc:
        print(f"[RESUME WARNING] Không đọc được epoch metadata từ {checkpoint_path}: {exc}")
        return 0

def stage_is_complete(checkpoint_path, requested_epochs):
    completed_epochs = checkpoint_epoch_number(checkpoint_path)
    print(f"[RESUME CHECK] checkpoint_epoch={completed_epochs}, target={requested_epochs}")
    return completed_epochs >= int(requested_epochs)

if RESUME_CKPT is None and AUTO_RESUME_FROM_STORE:
    direct_last = RUNS_ROOT / EXP_NAME / "weights" / "last.pt"
    if not IS_KAGGLE and direct_last.is_file():
        RESUME_CKPT = direct_last
        print("[DIRECT RESUME]", RESUME_CKPT)
    elif CHECKPOINT_STORE is not None:
        RESUME_CKPT = CHECKPOINT_STORE.for_run(EXP_NAME).download_latest(WORK_ROOT / "resume" / EXP_NAME)

BASE_COMPLETED_CKPT = None
if RESUME_CKPT:
    checkpoint = Path(RESUME_CKPT)
    assert checkpoint.exists(), checkpoint
    model = YOLO(str(checkpoint))
    if stage_is_complete(checkpoint, EPOCHS):
        BASE_COMPLETED_CKPT = checkpoint
        results = None
        print(f"[BASE COMPLETE] Bỏ qua train lại {EPOCHS} epoch; chuyển sang đánh giá.")
    else:
        attach_checkpoint_callbacks(model, EXP_NAME)
        results = model.train(
            data=str(yaml_path), resume=True, epochs=EPOCHS, time=BASE_TRAIN_HOURS,
            device=DEVICE, workers=WORKERS, cache=CACHE,
            save=True, save_period=SAVE_PERIOD, plots=True,
        )
else:
    model = YOLO(MODEL_SIZE)
    attach_checkpoint_callbacks(model, EXP_NAME)
    results = model.train(
        data=str(yaml_path), imgsz=IMG_SIZE, epochs=EPOCHS, time=BASE_TRAIN_HOURS, batch=BATCH,
        device=DEVICE, workers=WORKERS, patience=PATIENCE, cache=CACHE,
        project=str(RUNS_ROOT), name=EXP_NAME, exist_ok=True,
        save=True, save_period=SAVE_PERIOD, amp=True, plots=True,
        optimizer="auto", cos_lr=True, close_mosaic=10,
        degrees=4.0, translate=0.08, scale=0.45,
        mosaic=0.8, mixup=0.05, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.55, hsv_v=0.35,
        box=7.5, cls=0.5, dfl=1.5,
        seed=SEED, deterministic=True, verbose=True,
    )

print("Run:", save_dir)
print("Resume:", save_dir / "weights" / "last.pt")
"""
        ),
        code(
            r"""
# 10) XÁC ĐỊNH CHECKPOINT BASE
weights_dir = RUNS_ROOT / EXP_NAME / "weights"
best_pt = weights_dir / "best.pt"
last_pt = weights_dir / "last.pt"
remote_best = None
if not best_pt.exists() and CHECKPOINT_STORE is not None:
    remote_best = CHECKPOINT_STORE.for_run(EXP_NAME).download_named(
        "best.pt", WORK_ROOT / "resume" / EXP_NAME, required=False,
    )
fallback_checkpoint = BASE_COMPLETED_CKPT or (Path(RESUME_CKPT) if RESUME_CKPT else None)
checkpoint_candidates = [best_pt, remote_best, last_pt, fallback_checkpoint]
selected_checkpoint = next((Path(path) for path in checkpoint_candidates if path and Path(path).is_file()), None)
assert selected_checkpoint is not None, f"Không thấy checkpoint base hợp lệ trong {weights_dir} hoặc checkpoint store"
MODEL_FOR_INFER = str(selected_checkpoint)
print("best.pt:", best_pt.exists(), best_pt)
print("last.pt:", last_pt.exists(), last_pt)
print("MODEL_FOR_INFER:", MODEL_FOR_INFER)
"""
        ),
        code(
            r"""
# 11) ĐÁNH GIÁ TEST: mAP + PRECISION + RECALL + F1
import json
from ultralytics import YOLO

def evaluate_checkpoint(weights, run_name):
    detector = YOLO(str(weights))
    metrics = detector.val(
        data=str(yaml_path), split="test", imgsz=IMG_SIZE, batch=BATCH,
        device=DEVICE, workers=WORKERS, plots=True,
        project=str(RUNS_ROOT), name=run_name, exist_ok=True,
    )
    p_macro = float(metrics.box.mp)
    r_macro = float(metrics.box.mr)
    f1_macro = 2 * p_macro * r_macro / max(p_macro + r_macro, 1e-12)
    per_class = {}
    class_ids = [int(v) for v in metrics.box.ap_class_index]
    for pos, class_id in enumerate(class_ids):
        p = float(metrics.box.p[pos]); r = float(metrics.box.r[pos])
        per_class[detector.names[class_id]] = {
            "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "map50": float(metrics.box.ap50[pos]),
            "map50_95": float(metrics.box.ap[pos]),
        }
    summary = {
        "weights": str(weights), "precision_macro": p_macro,
        "recall_macro": r_macro, "f1_macro": f1_macro,
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }
    summary["target_met"] = summary["map50"] >= 0.85 and summary["f1_macro"] >= 0.85
    output = RUNS_ROOT / run_name / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

BASE_SUMMARY = evaluate_checkpoint(MODEL_FOR_INFER, f"{EXP_NAME}_test")
"""
        ),
    ]

    weak_stage = [
        markdown(
            r"""
## Giai đoạn 2 — weak supervision có kiểm soát

AUC v2 có khoảng 5.418 ảnh thuộc nhóm texting/talking-phone nhưng chỉ dùng khi người chạy có quyền/password hợp lệ. Teacher base chỉ giữ box `phone` có confidence cao. Tám ảnh Seatbelt Real chỉ giữ box `seatbelt` có confidence cao. Ảnh không có box đáng tin bị bỏ qua.
"""
        ),
        code(
            r"""
# 12) PSEUDO-LABEL AUC + SEATBELT REAL
PSEUDO_READY = False
PSEUDO_REPORT = None
combined_yaml = None

helper_candidates = list(training_code_dir.rglob("pseudo_label_weak_dms_sources.py"))
auc_zip = find_input_file(AUC_ARCHIVE_NAME, required=False)
seatbelt_zip = find_input_file(SEATBELT_ARCHIVE_NAME, required=False)
# Kaggle Upload Data có thể giải nén seatbelt ZIP thành thư mục. Helper pseudo-label
# cần ZipFile, nên đóng gói lại tám ảnh vào scratch disk nếu cần.
if seatbelt_zip is None:
    seatbelt_dir = find_input_dir("seatbelt_real_unlabelled", required=False)
    if seatbelt_dir is not None:
        seatbelt_images = sorted(
            path for path in seatbelt_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if seatbelt_images:
            seatbelt_zip = WORK_ROOT / "seatbelt_real_unlabelled_runtime.zip"
            with zipfile.ZipFile(seatbelt_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for image_path in seatbelt_images:
                    archive.write(image_path, image_path.relative_to(seatbelt_dir).as_posix())
            print(f"Repacked Kaggle seatbelt directory: {len(seatbelt_images)} images -> {seatbelt_zip}")
AUC_PASSWORD = os.getenv("AUC_ZIP_PASSWORD")
if IS_KAGGLE and not AUC_PASSWORD:
    try:
        from kaggle_secrets import UserSecretsClient
        AUC_PASSWORD = UserSecretsClient().get_secret("AUC_ZIP_PASSWORD")
    except Exception:
        AUC_PASSWORD = None

if PSEUDO_LABEL_ENABLED and helper_candidates and (auc_zip or seatbelt_zip):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dms_pseudo", helper_candidates[0])
    dms_pseudo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dms_pseudo)
    pseudo_dir = WORK_ROOT / "pseudo_labels"
    PSEUDO_REPORT = dms_pseudo.generate_pseudo_labels(
        weights=Path(MODEL_FOR_INFER),
        auc_zip=auc_zip or Path("/nonexistent/auc.zip"),
        seatbelt_zip=seatbelt_zip or Path("/nonexistent/seatbelt.zip"),
        base_yaml=yaml_path, output_dir=pseudo_dir, confidence=PSEUDO_CONF,
        image_size=IMG_SIZE, batch_size=16, device=str(DEVICE),
        limit=PSEUDO_LIMIT, overwrite=True, auc_password=AUC_PASSWORD,
    )
    accepted = sum(v.get("accepted_images", 0) for v in PSEUDO_REPORT["sources"].values())
    combined_yaml = Path(PSEUDO_REPORT["combined_yaml"])
    PSEUDO_READY = accepted > 0 and combined_yaml.exists()
    print("Pseudo accepted:", accepted, "ready:", PSEUDO_READY)
else:
    print("Bỏ qua pseudo-label: thiếu weak-source archive hoặc training_code.zip.")
"""
        ),
        code(
            r"""
# 13) FINE-TUNE VỚI PSEUDO-LABEL, GIỮ VAL/TEST GOLD GỐC
FINE_SUMMARY = None
if PSEUDO_READY:
    if FINE_RESUME_CKPT is None and AUTO_RESUME_FROM_STORE:
        direct_fine_last = RUNS_ROOT / FINE_TUNE_NAME / "weights" / "last.pt"
        if not IS_KAGGLE and direct_fine_last.is_file():
            FINE_RESUME_CKPT = direct_fine_last
        elif CHECKPOINT_STORE is not None:
            FINE_RESUME_CKPT = CHECKPOINT_STORE.for_run(FINE_TUNE_NAME).download_latest(
                WORK_ROOT / "resume" / FINE_TUNE_NAME,
            )

    fine_complete = bool(FINE_RESUME_CKPT) and stage_is_complete(FINE_RESUME_CKPT, PSEUDO_EPOCHS)
    if fine_complete:
        print(f"[FINE COMPLETE] Bỏ qua train lại {PSEUDO_EPOCHS} epoch; chuyển sang đánh giá.")
    elif FINE_RESUME_CKPT:
        fine_model = YOLO(str(FINE_RESUME_CKPT))
        attach_checkpoint_callbacks(fine_model, FINE_TUNE_NAME)
        fine_model.train(
            data=str(combined_yaml), resume=True, epochs=PSEUDO_EPOCHS, time=FINE_TRAIN_HOURS,
            device=DEVICE, workers=WORKERS, cache=False,
            save=True, save_period=SAVE_PERIOD, plots=True,
        )
    else:
        fine_model = YOLO(MODEL_FOR_INFER)
        attach_checkpoint_callbacks(fine_model, FINE_TUNE_NAME)
        fine_model.train(
            data=str(combined_yaml), imgsz=IMG_SIZE, epochs=PSEUDO_EPOCHS,
            time=FINE_TRAIN_HOURS, batch=BATCH,
            device=DEVICE, workers=WORKERS, patience=10, cache=False,
            project=str(RUNS_ROOT), name=FINE_TUNE_NAME, exist_ok=True,
            save=True, save_period=SAVE_PERIOD, amp=True, plots=True,
            optimizer="AdamW", lr0=0.002, lrf=0.05, cos_lr=True, close_mosaic=5,
            mosaic=0.4, mixup=0.0, fliplr=0.5,
            seed=SEED, deterministic=True,
        )

    fine_local_best = RUNS_ROOT / FINE_TUNE_NAME / "weights" / "best.pt"
    fine_remote_best = None
    if not fine_local_best.exists() and CHECKPOINT_STORE is not None:
        fine_remote_best = CHECKPOINT_STORE.for_run(FINE_TUNE_NAME).download_named(
            "best.pt", WORK_ROOT / "resume" / FINE_TUNE_NAME, required=False,
        )
    fine_candidates = [fine_local_best, fine_remote_best, FINE_RESUME_CKPT]
    fine_best = next((Path(path) for path in fine_candidates if path and Path(path).is_file()), None)
    assert fine_best is not None, "Không tìm thấy fine-tune checkpoint để đánh giá"
    FINE_SUMMARY = evaluate_checkpoint(fine_best, f"{FINE_TUNE_NAME}_test")
else:
    print("Không có pseudo-label đạt ngưỡng; giữ base checkpoint.")
"""
        ),
        code(
            r"""
# 14) CHỌN CHAMPION + EXPORT PT/ONNX + GÓI OUTPUT KAGGLE
import shutil

def conservative_score(summary):
    return min(float(summary["map50"]), float(summary["f1_macro"]))

CHAMPION_SUMMARY = BASE_SUMMARY
if FINE_SUMMARY and conservative_score(FINE_SUMMARY) > conservative_score(BASE_SUMMARY):
    CHAMPION_SUMMARY = FINE_SUMMARY

MODEL_FOR_INFER = CHAMPION_SUMMARY["weights"]
champion = YOLO(MODEL_FOR_INFER)
artifact_dir = RUNS_ROOT / "champion_artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(MODEL_FOR_INFER, artifact_dir / "best.pt")
onnx_path = Path(str(champion.export(format="onnx", imgsz=IMG_SIZE, simplify=True, dynamic=True)))
if onnx_path.exists():
    shutil.copy2(onnx_path, artifact_dir / "best.onnx")
(artifact_dir / "metrics_summary.json").write_text(
    json.dumps(CHAMPION_SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8"
)
archive = shutil.make_archive(str(RUNS_ROOT / "dms_champion_artifacts"), "zip", artifact_dir)
if CHECKPOINT_STORE is not None:
    champion_store = CHECKPOINT_STORE.for_run("champion_artifacts")
    for artifact in [artifact_dir / "best.pt", artifact_dir / "best.onnx", artifact_dir / "metrics_summary.json", Path(archive)]:
        if artifact.is_file():
            champion_store.upload(artifact)
print("Champion:", json.dumps(CHAMPION_SUMMARY, ensure_ascii=False, indent=2))
print("Kaggle Output:", archive)
print("MODEL_FOR_INFER:", MODEL_FOR_INFER)
"""
        ),
    ]

    # Refresh legacy prose while retaining the proven image/video inference code.
    inference_tail[-2] = markdown(
        r"""
# Resume train trên Kaggle / Colab

- Kaggle: mặc định `AUTO_RESUME_FROM_STORE=True`; notebook tải `epoch_NNN.pt` mới nhất trong thư mục run trên Drive và kiểm tra size/MD5 trước khi resume.
- Nếu muốn chọn thủ công, tải checkpoint về hoặc attach như Kaggle Dataset/Model input rồi đặt `RESUME_CKPT` tới file đó.
- Colab: đặt `last.pt` trong Google Drive và gán đường dẫn vào `RESUME_CKPT`.
- Không resume từ `best.pt` nếu cần giữ optimizer/epoch state.
- `checkpoint_sync_ready.json` phải xuất hiện trong folder Drive trước khi cell train chạy.
- Mỗi epoch tạo `epoch_NNN.pt`; `best.pt`, `last.pt`, `results.csv`, `args.yaml` và `checkpoint_latest.json` được đồng bộ có xác minh.
- Base/fine-tune đã hoàn thành sẽ được nhận diện để tránh lỗi Ultralytics `nothing to resume`.
- **Không** đổi `RUNS_ROOT` thành đường dẫn có chữ `drive` trên Kaggle: đó vẫn chỉ là thư mục output cục bộ và sẽ mất khi session hết hạn.
"""
    )
    inference_tail[-1] = markdown(
        r"""
# Gợi ý trình bày trong khóa luận

- Báo cáo rõ ba tầng: YOLO11 detector → MediaPipe Pose context → temporal smoothing.
- Ghi nguồn/mapping nhãn riêng cho từng dataset; AUC là weak-label, không mô tả như bounding-box ground truth.
- Báo cáo cả Precision, Recall, macro F1, mAP@50 và mAP@50–95 trên test group-disjoint.
- Chỉ kết luận đạt mục tiêu >85% khi `metrics_summary.json` có `target_met: true`; không suy diễn từ train loss hoặc một vài video demo.
"""
    )

    notebook["cells"] = front + weak_stage + inference_tail
    notebook.setdefault("metadata", {})["dms_pipeline_version"] = "4class-multisource-rtx5090-5h-v9"
    notebook["metadata"]["accelerator"] = "GPU"
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
