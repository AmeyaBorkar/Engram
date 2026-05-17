"""Manifest model and writer.

Every benchmark run produces a JSON manifest in `benchmarks/runs/`. The
manifest is the unit of evidence in `benchmarks/SCOREBOARD.md`: claims
without one don't count.
"""

# Best-effort environment capture invokes git/sysctl from PATH; that's
# intentional and applies file-wide.
# ruff: noqa: S607

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import platform
import secrets
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """One benchmark run, captured for reproducibility."""

    suite: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    python_version: str
    os: str
    cpu: str
    ram_gb: float
    provider: str
    provider_hash: str
    dataset_version: str
    dataset_checksum: str
    engram_config: dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, list[float]] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, list[float]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str, sort_keys=True)

    def write(self, runs_dir: Path) -> Path:
        """Write the manifest to `runs_dir` atomically. Returns the file path.

        H-75 fix: ``Path.write_text`` is not atomic and the original
        filename used second-precision timestamps, so two runs in the
        same second silently overwrote each other's evidence. The
        replacement strategy:

        1. Build the base filename (timestamp-sha-suite).
        2. If a manifest of that exact name already exists, append a
           4-hex-char random salt suffix and try again until we find a
           fresh path. The salt has 16 bits of entropy per attempt; a
           collision would need two writes to draw the same 4-hex
           value AND beat the existence check in the same second.
        3. Serialize JSON to a temp file in the same directory, then
           ``os.replace(tmp, dest)`` -- atomic on POSIX and on Windows
           when both paths sit on the same volume. A crash mid-write
           leaves the temp file behind (visible, easy to clean up)
           rather than a half-written manifest.
        """
        runs_dir.mkdir(parents=True, exist_ok=True)
        sha = (self.git_commit[:8] or "nogit") + ("-dirty" if self.git_dirty else "")
        ts = self.timestamp.replace(":", "").replace("-", "").replace(".", "_")
        base = f"{ts}-{sha}-{self.suite}"
        path = runs_dir / f"{base}.json"
        # Same-second collisions get a salt suffix until we find a
        # filename that doesn't already exist. The TOCTOU window
        # between exists() and replace() is closed below by the atomic
        # os.replace.
        while path.exists():
            salt = secrets.token_hex(2)  # 4 hex chars
            path = runs_dir / f"{base}-{salt}.json"
        payload = self.to_json()
        # NamedTemporaryFile with delete=False so we control the
        # rename ourselves; suffix matches the destination so a glob
        # for *.json still sees it if the process crashes mid-write.
        fd, tmp_name = tempfile.mkstemp(prefix=f"{base}.", suffix=".json.tmp", dir=str(runs_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            # Best-effort cleanup: leave nothing behind on failure.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        return path


def gather_environment() -> dict[str, Any]:
    """Capture environment info for a manifest. Best-effort across platforms."""
    return {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": _ram_gb(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip()) if result.returncode == 0 else False


def _ram_gb() -> float:
    """Approximate installed RAM in GB. Returns 0.0 if unable to determine."""
    try:
        if sys.platform == "win32":
            buf = ctypes.c_uint64(0)
            ok = ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(buf))
            if ok:
                return round(buf.value / (1024 * 1024), 2)
        elif sys.platform == "linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 2)
        elif sys.platform == "darwin":
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if r.returncode == 0:
                return round(int(r.stdout.strip()) / (1024**3), 2)
    except (OSError, ValueError, AttributeError):
        return 0.0
    return 0.0


def manifest_from_run(
    *,
    suite_name: str,
    provider_name: str,
    provider_hash: str,
    dataset_version: str,
    dataset_checksum: str,
    aggregate_metrics: dict[str, float],
    confidence_intervals: dict[str, tuple[float, float]],
    per_question: list[dict[str, Any]],
    latency_ms: dict[str, list[float]],
    engram_config: dict[str, Any] | None = None,
) -> Manifest:
    """Convenience constructor that captures the environment and timestamp."""
    env = gather_environment()
    return Manifest(
        suite=suite_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=env["git_commit"],
        git_dirty=env["git_dirty"],
        python_version=env["python_version"],
        os=env["os"],
        cpu=env["cpu"],
        ram_gb=env["ram_gb"],
        provider=provider_name,
        provider_hash=provider_hash,
        dataset_version=dataset_version,
        dataset_checksum=dataset_checksum,
        engram_config=dict(engram_config or {}),
        aggregate_metrics=dict(aggregate_metrics),
        confidence_intervals={k: list(v) for k, v in confidence_intervals.items()},
        per_question=list(per_question),
        latency_ms={k: list(v) for k, v in latency_ms.items()},
    )
