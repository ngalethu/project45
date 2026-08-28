from __future__ import annotations
from pathlib import Path
from typing import Optional
import shutil
from fastapi import UploadFile

from app.common.config import load_config
from app.common.utils import ensure_dir

import re
import time
from app.common.config import load_config
from app.common.utils import ensure_dir

cfg = load_config()
BASE_UPLOAD_DIR = ensure_dir(cfg["storage"]["cloud_upload_dir"])

def sanitize_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    base = Path(filename).stem
    clean_base = re.sub(r"[^\w\-]", "_", base)
    if not clean_base:
        clean_base = "upload"
    return f"{clean_base}{ext}"

def save_upload(upload: Optional[UploadFile], subdir: str) -> str | None:
    if upload is None:
        return None

    target_dir = ensure_dir(BASE_UPLOAD_DIR / subdir)

    raw_filename = upload.filename or "upload.bin"
    safe_name = sanitize_filename(raw_filename)
    timestamp = int(time.time() * 1000)
    target_filename = f"{timestamp}_{safe_name}"

    target_path = target_dir / target_filename

    with open(target_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    return str(target_path).replace("\\", "/")