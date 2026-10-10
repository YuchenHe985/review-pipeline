"""Outputs are published atomically: written into a staging folder, then renamed into place.

A crash, a kill or an exception leaves either the complete previous state or the complete new
output, never a half-written folder that a later step would read. Each output folder carries a
manifest.json; a folder without one is not an output.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def read_manifest(folder: Path) -> dict[str, Any] | None:
    p = folder / MANIFEST
    return json.loads(p.read_text()) if p.is_file() else None


@contextlib.contextmanager
def publish(final: Path) -> Iterator[Path]:
    """Yields a staging folder; on success it becomes `final`. On any exception it is removed."""
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".staging-{final.name}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        yield staging
        if not (staging / MANIFEST).is_file():
            raise RuntimeError(f"refusing to publish {final.name} without a manifest")
        if final.exists():  # a concurrent run published the same id first; identical content by construction
            shutil.rmtree(staging)
            return
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_manifest(folder: Path, body: dict[str, Any]) -> None:
    body = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), **body}
    (folder / MANIFEST).write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")


def sweep_staging(root: Path) -> int:
    """Removes staging folders left by a killed process. Returns how many were removed."""
    n = 0
    if root.is_dir():
        for p in root.rglob(".staging-*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                n += 1
    return n
