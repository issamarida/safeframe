"""Model weight management.

Weights are never shipped in the repository. They are downloaded on first use
into a cache directory (``SAFEFRAME_MODEL_DIR``, default
``~/.cache/safeframe/models``) and verified against a pinned SHA-256 where the
upstream file is immutable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default person model for both the CLI and the server: YOLO11 nano instance
# segmentation (boxes for tracking + a silhouette per person). Downloaded by
# Ultralytics into model_dir() on first use; never committed. Override with
# ``--yolo`` or ``SAFEFRAME_YOLO_WEIGHTS``; detection-only checkpoints such as
# "yolo11n.pt" remain supported and fall back to rectangular blur.
DEFAULT_YOLO_WEIGHTS = "yolo11n-seg.pt"


def model_dir() -> Path:
    root = os.environ.get("SAFEFRAME_MODEL_DIR") or Path.home() / ".cache" / "safeframe" / "models"
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    url: str
    sha256: str | None = None


YUNET = RemoteFile(
    filename="face_detection_yunet_2023mar.onnx",
    url=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(remote: RemoteFile) -> Path:
    """Return a local path to ``remote``, downloading and verifying it if needed."""
    dest = model_dir() / remote.filename
    if dest.exists() and (remote.sha256 is None or _sha256(dest) == remote.sha256):
        return dest
    log.info("downloading %s", remote.url)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(remote.url, tmp_path)
        if remote.sha256 is not None:
            actual = _sha256(tmp_path)
            if actual != remote.sha256:
                raise RuntimeError(
                    f"checksum mismatch for {remote.filename}: "
                    f"expected {remote.sha256}, got {actual}"
                )
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest


def yolo_weights_path(name: str) -> Path:
    """Path inside the cache where Ultralytics will download ``name`` on first use."""
    return model_dir() / name
