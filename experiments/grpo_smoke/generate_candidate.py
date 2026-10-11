#!/usr/bin/env python3
"""Generate exactly one evaluator-ready candidate from a pinned Qwen checkpoint.

The normal path loads one immutable Hugging Face revision and uses greedy
decoding. ``--validate-only`` checks the task contract and prints the resolved
prompt/settings without importing the ML stack or downloading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SCHEMA_VERSION = "sysllmatic.attempt.v1"
SEED = 42
MAX_NEW_TOKENS = 2048
FENCE_PATTERN = re.compile(r"```(?:[A-Za-z0-9_+.-]+)?[ \t]*\r?\n(.*?)```", re.DOTALL)
LANGUAGE_EXTENSIONS = {
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "cuda": ".cu",
    "python": ".py",
    "py": ".py",
    "triton": ".py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_task(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    task = json.loads(raw)
    required_strings = ("task_id", "benchmark", "language", "description", "reference_implementation")
    for field in required_strings:
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"task field {field!r} must be a non-empty string")
    if "constraints" in task and not (
        isinstance(task["constraints"], list)
        and all(isinstance(item, str) and item.strip() for item in task["constraints"])
    ):
        raise ValueError("task field 'constraints' must be a list of non-empty strings")
    if "hardware" in task and not isinstance(task["hardware"], dict):
        raise ValueError("task field 'hardware' must be an object")
    if "interface" in task and not isinstance(task["interface"], dict):
        raise ValueError("task field 'interface' must be an object")
    return task, raw


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    constraints = task.get("constraints", [])
    constraint_text = "\n".join(f"- {item}" for item in constraints) or "- Preserve the supplied interface."
    hardware = json.dumps(task.get("hardware", {}), indent=2, sort_keys=True)
    interface = json.dumps(task.get("interface", {}), indent=2, sort_keys=True)
    language = task["language"]
    system = (
        "You produce one candidate implementation for execution-based evaluation. "
        "Optimize the supplied reference while preserving its exact semantics and interface. "
        "Respect every hardware and task constraint. Return exactly one complete source file "
        f"in a single {language} fenced code block and no other text. Do not invent benchmark results."
    )
    user = (
        f"Task ID: {task['task_id']}\n"
        f"Benchmark: {task['benchmark']}\n"
        f"Goal: {task['description']}\n\n"
        f"Hardware:\n{hardware}\n\n"
        f"Required interface:\n{interface}\n\n"
        f"Constraints:\n{constraint_text}\n\n"
        f"Reference implementation:\n```{language}\n{task['reference_implementation'].rstrip()}\n```\n\n"
        "Return the single complete candidate now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fixed_generation_settings() -> dict[str, Any]:
    return {
        "decoding_strategy": "greedy",
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 1,
        "num_return_sequences": 1,
        "seed": SEED,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }


def extract_candidate(raw_completion: str) -> tuple[str, dict[str, Any]]:
    blocks = FENCE_PATTERN.findall(raw_completion)
    warnings: list[str] = []
    if blocks:
        code = blocks[0].strip() + "\n"
        method = "first_fenced_block"
        if len(blocks) > 1:
            warnings.append(f"model returned {len(blocks)} fenced blocks; only the first was retained")
    else:
        code = raw_completion.strip() + "\n"
        method = "unfenced_full_completion"
        warnings.append("model returned no fenced code block; the full completion was retained")
    if not code.strip():
        raise ValueError("model returned an empty candidate")
    return code, {"method": method, "fence_count": len(blocks), "warnings": warnings}


def candidate_filename(task: dict[str, Any]) -> str:
    requested = task.get("interface", {}).get("candidate_filename")
    if requested is None:
        extension = LANGUAGE_EXTENSIONS.get(task["language"].strip().lower(), ".txt")
        return f"candidate{extension}"
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("interface.candidate_filename must be a non-empty string")
    requested = requested.strip()
    if (
        requested in {".", ".."}
        or Path(requested).name != requested
        or "/" in requested
        or "\\" in requested
    ):
        raise ValueError("interface.candidate_filename must be a basename, not a path")
    return requested


def directory_manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"adapter directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path, help="PKB task contract JSON")
    parser.add_argument("--output-dir", type=Path, help="directory for candidate.py and attempt.json")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="optional LoRA adapter from the five-step run; base checkpoint is used when omitted",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="fail instead of downloading the pinned model when it is absent from the cache",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when CUDA is unavailable instead of falling back to recorded CPU generation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the task and print immutable settings without loading the model",
    )
    args = parser.parse_args()
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    return args


def main() -> int:
    args = parse_args()
    task_path = args.task.resolve()
    task, task_raw = load_task(task_path)
    output_candidate_name = candidate_filename(task)
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    if adapter_path is not None and not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter path is not a directory: {adapter_path}")
    adapter_manifest = directory_manifest_sha256(adapter_path) if adapter_path else None
    messages = build_messages(task)
    settings = fixed_generation_settings()
    settings["device_policy"] = "cuda_required" if args.require_cuda else "auto"
    prompt_contract = {"messages": messages, "chat_template": "checkpoint tokenizer chat template"}
    prompt_contract_sha256 = sha256_text(stable_json(prompt_contract))
    task_sha256 = sha256_text(task_raw)
    identity = {
        "generation": settings,
        "adapter_manifest_sha256": adapter_manifest,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_contract_sha256": prompt_contract_sha256,
        "task_sha256": task_sha256,
    }
    attempt_id = sha256_text(stable_json(identity))[:16]

    validation = {
        "attempt_id": attempt_id,
        "candidate_filename": output_candidate_name,
        "generation": settings,
        "adapter_manifest_sha256": adapter_manifest,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_contract_sha256": prompt_contract_sha256,
        "task_id": task["task_id"],
        "task_sha256": task_sha256,
        "valid": True,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    # Set deterministic CUDA requirements before importing torch.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    try:
        import peft
        from peft import PeftModel
    except ImportError:
        peft = None
        PeftModel = None

    set_seed(SEED)
    torch.use_deterministic_algorithms(True)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable and --require-cuda was requested")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    if adapter_path is not None and PeftModel is None:
        raise RuntimeError("peft is required when --adapter-path is supplied")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_copy_path = output_dir / "task.json"
    task_copy_path.write_text(task_raw, encoding="utf-8", newline="\n")
    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
        )
    ).resolve()
    if snapshot_path.name != MODEL_REVISION:
        raise RuntimeError(
            f"expected snapshot {MODEL_REVISION}, but Hugging Face resolved {snapshot_path.name}"
        )

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.to(device)
    model.eval()

    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_tokens = int(encoded["input_ids"].shape[1])
    context_limit = int(getattr(model.config, "max_position_embeddings", 32768))
    if input_tokens + MAX_NEW_TOKENS > context_limit:
        raise ValueError(
            f"prompt ({input_tokens} tokens) plus completion budget ({MAX_NEW_TOKENS}) "
            f"exceeds model context ({context_limit})"
        )

    started_at = utc_now()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed_seconds = time.perf_counter() - started
    generated_ids = generated[0, input_tokens:]
    raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    candidate_code, extraction = extract_candidate(raw_completion)

    candidate_name = output_candidate_name
    candidate_path = output_dir / candidate_name
    raw_path = output_dir / "raw_completion.txt"
    prompt_path = output_dir / "rendered_prompt.txt"
    candidate_path.write_text(candidate_code, encoding="utf-8", newline="\n")
    raw_path.write_text(raw_completion, encoding="utf-8", newline="\n")
    prompt_path.write_text(rendered_prompt, encoding="utf-8", newline="\n")

    record = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "created_at_utc": started_at,
        "task": {
            "id": task["task_id"],
            "benchmark": task["benchmark"],
            "language": task["language"],
            "source_path": task_copy_path.name,
            "input_path": str(task_path),
            "source_sha256": task_sha256,
            "description": task["description"],
            "hardware": task.get("hardware", {}),
            "interface": task.get("interface", {}),
            "constraints": task.get("constraints", []),
        },
        "model": {
            "provider": "huggingface",
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_path": str(snapshot_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "adapter_manifest_sha256": adapter_manifest,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
        },
        "generation": {
            **settings,
            "status": "completed",
            "prompt_contract_sha256": prompt_contract_sha256,
            "rendered_prompt_path": prompt_path.name,
            "rendered_prompt_sha256": sha256_text(rendered_prompt),
            "input_tokens": input_tokens,
            "output_tokens": int(generated_ids.shape[0]),
            "elapsed_seconds": elapsed_seconds,
        },
        "candidate": {
            "path": candidate_name,
            "sha256": sha256_text(candidate_code),
            "raw_completion_path": raw_path.name,
            "raw_completion_sha256": sha256_text(raw_completion),
            "extraction": extraction,
        },
        "evaluation": {
            "compilation": {"status": "not_run", "exit_code": None, "elapsed_seconds": None, "log": None},
            "execution": {"status": "not_run", "exit_code": None, "elapsed_seconds": None, "log": None},
            "correctness": {"status": "not_run", "passed": None, "details": None},
            "timing": {
                "status": "not_run",
                "reference_seconds": None,
                "candidate_seconds": None,
                "speedup": None,
                "samples": None,
            },
            "feedback": {"status": "not_available", "text": None, "source": None},
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__ if peft is not None else None,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    record_path = output_dir / "attempt.json"
    write_json(record_path, record)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "candidate": str(candidate_path),
                "record": str(record_path),
                "model_revision": MODEL_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
