#!/usr/bin/env python3
"""Copy audit-sized run artifacts into Git without copying model weights."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


AUDIT_FILES = (
    "summary.json",
    "config.json",
    "metrics.json",
    "samples.json",
    "environment.txt",
    "run.log",
)

OPTIONAL_AUDIT_FILES = (
    "REPORT.md",
    "nvidia-smi.txt",
    "runner.log",
    "runner_status.json",
    "slurm-sacct-final.txt",
    "slurm-sacct-in-job.txt",
    "slurm-sacct.txt",
    "source_manifest.sha256",
    "system-environment.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    for filename in AUDIT_FILES:
        source = run_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / filename)

    for filename in OPTIONAL_AUDIT_FILES:
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)

    checkpoints = sorted((run_dir / "trainer_output").glob("checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError("run contains no checkpoint")
    checkpoint = checkpoints[-1]
    manifest_lines = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(checkpoint).as_posix()}")
    (destination / "checkpoint_manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    total_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    size_record = f"{total_bytes} bytes\n"
    (destination / "artifact_size.txt").write_text(size_record, encoding="utf-8")
    # Keep the original filename so existing CPU preflight automation does not
    # break. New callers should use the platform-neutral artifact_size.txt.
    (destination / "wsl_artifact_size.txt").write_text(size_record, encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
