#!/usr/bin/env python3
"""Fail if footage, model weights, or likely secrets are about to be committed.

SafeFrame processes video of children. Footage must never reach version
control, and neither should model weights (licensing, size) or credentials.
This check runs as a pre-commit hook (on staged files) and in CI (on every
tracked file), so a mistake is caught even if a contributor skips the hook.

Usage:
    check_forbidden_files.py [FILE ...]   # check the given paths
    check_forbidden_files.py --all        # check every file tracked by git
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FOOTAGE_EXT = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv",
    ".mpg", ".mpeg", ".3gp", ".mts", ".m2ts", ".wav", ".mp3", ".aac", ".m4a",
}
WEIGHT_EXT = {
    ".pt", ".pth", ".onnx", ".engine", ".tflite", ".caffemodel",
    ".safetensors", ".bin", ".ckpt", ".h5", ".pb",
}
# Silhouette mask stores outline real people: derived personal data.
MASK_EXT = {".masks", ".sfm", ".partial"}
SECRET_NAMES = re.compile(r"(^|/)(\.env(\..+)?|.*\.pem|.*\.key|id_rsa|id_ed25519)$")
ALLOWED_SECRET_NAMES = {".env.example"}

SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub tokens
    re.compile(rb"hf_[A-Za-z0-9]{30,}"),  # Hugging Face tokens
    re.compile(rb"sk-[A-Za-z0-9_-]{32,}"),  # generic API secret keys
]

MAX_BYTES = 2 * 1024 * 1024  # anything larger than 2 MiB needs a very good reason


def problems_for(path: Path) -> list[str]:
    name = path.as_posix()
    suffix = path.suffix.lower()
    found: list[str] = []

    if suffix == ".ts" and name.startswith("frontend/"):
        pass  # TypeScript source, not an MPEG transport stream
    elif suffix in FOOTAGE_EXT or suffix == ".ts":
        found.append("video/audio footage must never be committed")

    if suffix in MASK_EXT:
        found.append("silhouette mask stores are derived from footage and must never be committed")

    if suffix in WEIGHT_EXT:
        found.append("model weights must never be committed (they are downloaded at runtime)")

    if SECRET_NAMES.search(name) and path.name not in ALLOWED_SECRET_NAMES:
        found.append("looks like a secrets/credentials file")

    if path.is_file():
        size = path.stat().st_size
        if size > MAX_BYTES:
            found.append(f"file is {size / 1e6:.1f} MB (limit {MAX_BYTES / 1e6:.0f} MB)")
        if size <= MAX_BYTES and path.resolve() != Path(__file__).resolve():
            data = path.read_bytes()
            for pattern in SECRET_PATTERNS:
                if pattern.search(data):
                    found.append(f"contains something that looks like a secret ({pattern.pattern[:24]!r}...)")
                    break
    return found


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def main(argv: list[str]) -> int:
    paths = tracked_files() if argv == ["--all"] else [Path(a) for a in argv]
    failures = 0
    for path in paths:
        for problem in problems_for(path):
            print(f"FORBIDDEN  {path}: {problem}", file=sys.stderr)
            failures += 1
    if failures:
        print(
            f"\n{failures} problem(s) found. Footage, weights and secrets stay out of git — "
            "see CONTRIBUTING.md#data-handling.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
