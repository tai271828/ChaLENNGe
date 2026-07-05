"""Provenance recording shared by training and evaluation drivers.

Every result directory must carry a manifest.json that makes the run
reproducible (context-plan-guidance/00-README.md, rule 1).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    """Current repo commit hash, or None if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256(path: str | Path) -> str:
    """SHA256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: str | Path, payload: dict) -> Path:
    """Write manifest.json into out_dir, prepending timestamp/command/commit."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        **payload,
    }
    path = Path(out_dir) / "manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path
