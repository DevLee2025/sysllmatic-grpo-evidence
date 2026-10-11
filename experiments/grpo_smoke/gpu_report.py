#!/usr/bin/env python3
"""Write a concise, auditable report for a Scholar GPU smoke-test run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--runner-elapsed-seconds", required=True, type=float)
    parser.add_argument("--python-module", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not recorded"
    return f"{value / (1024**3):.2f} GiB"


def seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "not recorded"
    return f"{value:.2f} s"


def text(value: Any) -> str:
    return "not recorded" if value in (None, "") else str(value)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = read_json(run_dir / "summary.json")
    summary_passed = bool(summary and summary.get("passed"))
    status = "PASS" if args.exit_code == 0 and summary_passed else "FAIL"

    runner_status = {
        "exit_code": args.exit_code,
        "python_module": args.python_module,
        "runner_elapsed_seconds": args.runner_elapsed_seconds,
        "status": status,
        "summary_present": summary is not None,
    }
    (run_dir / "runner_status.json").write_text(
        json.dumps(runner_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if summary is None:
        details = {
            "completed_steps": "not recorded",
            "max_steps": 5,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "device": "not recorded",
            "training_peak": "not recorded",
            "overall_peak": "not recorded",
            "training_time": "not recorded",
            "python_time": "not recorded",
            "checkpoint": "not written",
            "python": "not recorded",
            "torch": "not recorded",
            "cuda": "not recorded",
            "trl": "not recorded",
            "transformers": "not recorded",
            "job_id": "not recorded",
            "node_list": "not recorded",
            "partition": "not recorded",
        }
    else:
        versions = summary.get("versions") or {}
        slurm = summary.get("slurm") or {}
        details = {
            "completed_steps": summary.get("completed_steps"),
            "max_steps": summary.get("max_steps"),
            "model": summary.get("model_id"),
            "revision": summary.get("model_revision"),
            "device": summary.get("device"),
            "training_peak": gib(summary.get("training_peak_memory_allocated_bytes")),
            "overall_peak": gib(summary.get("max_memory_allocated_bytes")),
            "training_time": seconds(summary.get("training_elapsed_seconds")),
            "python_time": seconds(summary.get("python_elapsed_seconds")),
            "checkpoint": summary.get("reload_checkpoint_path"),
            "python": summary.get("python"),
            "torch": versions.get("torch"),
            "cuda": summary.get("cuda_runtime"),
            "trl": versions.get("trl"),
            "transformers": versions.get("transformers"),
            "job_id": slurm.get("job_id"),
            "node_list": slurm.get("node_list"),
            "partition": slurm.get("partition"),
        }

    lines = [
        "# Scholar A40 GRPO infrastructure check",
        "",
        f"**Status: {status}.** This is a training-infrastructure check, not evidence that GRPO improves code optimization.",
        "",
        "## Result",
        "",
        f"- Five-step run: {text(details['completed_steps'])}/{text(details['max_steps'])} optimizer steps completed",
        f"- Model: `{text(details['model'])}` at `{text(details['revision'])}`",
        f"- GPU: {text(details['device'])}",
        f"- Peak allocated GPU memory: {text(details['overall_peak'])} (training peak: {text(details['training_peak'])})",
        f"- Wall time: {args.runner_elapsed_seconds:.2f} s end to end; training {text(details['training_time'])}; Python run {text(details['python_time'])}",
        f"- Reloaded checkpoint: `{text(details['checkpoint'])}`",
        "",
        "## Environment",
        "",
        f"- Slurm: job `{text(details['job_id'])}`, node `{text(details['node_list'])}`, partition `{text(details['partition'])}`",
        f"- Python module: `{args.python_module}`",
        f"- Python: `{text(details['python'])}`",
        f"- PyTorch/CUDA: `{text(details['torch'])}` / `{text(details['cuda'])}`",
        f"- Transformers/TRL: `{text(details['transformers'])}` / `{text(details['trl'])}`",
        "- When produced, full records include `summary.json`, `config.json`, `metrics.json`, `samples.json`, `environment.txt`, `system-environment.txt`, `nvidia-smi.txt`, `source_manifest.sha256`, and logs in this directory.",
        "",
    ]
    if status == "FAIL":
        lines.extend(
            [
                "## Failure record",
                "",
                f"- Runner exit code: `{args.exit_code}`",
                "- Inspect `runner.log` for setup failures and `run.log`, when present, for trainer output.",
                "",
            ]
        )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
