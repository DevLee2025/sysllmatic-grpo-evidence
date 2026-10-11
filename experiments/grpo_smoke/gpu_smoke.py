#!/usr/bin/env python3
"""Five-step single-GPU GRPO smoke test using a real arithmetic reward."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import accelerate
import datasets
import peft
import torch
import transformers
import trl
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
GROUP_SIZE = 4
MAX_STEPS = 5
ANSWER_PATTERN = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE)
ROLLOUTS: list[dict[str, Any]] = []

PROBLEMS = (
    ("17 + 26", "43"),
    ("58 - 19", "39"),
    ("12 * 7", "84"),
    ("144 / 12", "12"),
    ("23 * 17", "391"),
    ("38 * 6 - 17", "211"),
    ("945 / 9", "105"),
    ("2^8", "256"),
    ("gcd(84, 126)", "42"),
    ("lcm(12, 18)", "36"),
    ("the sum of the integers from 1 through 20", "210"),
    ("15 percent of 260", "39"),
    ("x in the equation 3x + 7 = 52", "15"),
    ("7^2 + 5^2", "74"),
    ("1234 modulo 17", "10"),
    ("37 * 41 - 58", "1459"),
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().contiguous().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def arithmetic_reward(
    completions: list[Any],
    answer: list[str],
    completion_ids: list[Any] | None = None,
    **_: Any,
) -> list[float]:
    ids = completion_ids if completion_ids is not None else [[] for _ in completions]
    rewards: list[float] = []
    for completion, token_ids, expected in zip(completions, ids, answer, strict=True):
        text = completion_text(completion)
        matches = ANSWER_PATTERN.findall(text)
        formatted = len(matches) == 1
        correct = formatted and matches[0] == str(expected)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        token_list = [int(value) for value in token_ids]
        brevity = 0.1 * max(0.0, 1.0 - len(token_list) / 64.0) if correct else 0.0
        reward = (1.0 if correct else 0.0) + (0.1 if formatted else 0.0) + brevity
        rewards.append(reward)
        ROLLOUTS.append(
            {
                "answer": str(expected),
                "completion": text,
                "completion_ids": token_list,
                "correct": correct,
                "format_valid": formatted,
                "reward": reward,
            }
        )
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    program_started = time.perf_counter()
    args = parse_args()
    output_dir = args.output_dir.resolve()
    trainer_output = output_dir / "trainer_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the GPU smoke test")
    if trl.__version__ != "1.14.1":
        raise RuntimeError(f"expected TRL 1.14.1, found {trl.__version__}")

    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"Compute {problem}. Show concise reasoning and finish with "
                            "exactly <answer>INTEGER</answer>."
                        ),
                    }
                ],
                "answer": answer,
            }
            for problem, answer in PROBLEMS
        ]
    )

    snapshot_path = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    ).resolve()
    resolved_revision = snapshot_path.name if snapshot_path.parent.name == "snapshots" else None
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            f"expected model revision {MODEL_REVISION}, resolved {resolved_revision!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    config = GRPOConfig(
        output_dir=str(trainer_output),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=GROUP_SIZE,
        gradient_accumulation_steps=1,
        num_generations=GROUP_SIZE,
        max_completion_length=64,
        learning_rate=5e-5,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        temperature=1.0,
        use_cpu=False,
        use_vllm=False,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_steps=1,
        logging_strategy="steps",
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_steps=MAX_STEPS,
        save_strategy="steps",
        save_total_limit=1,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=arithmetic_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    initial_state = trainable_state(trainer.model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    training_elapsed_seconds = time.perf_counter() - started
    training_peak_allocated_bytes = torch.cuda.max_memory_allocated()
    training_peak_reserved_bytes = torch.cuda.max_memory_reserved()
    final_state = trainable_state(trainer.model)
    changed_tensors = [
        name for name, before in initial_state.items() if not torch.equal(before, final_state[name])
    ]
    initial_digest = state_digest(initial_state)
    final_digest = state_digest(final_state)
    metric_records = list(trainer.state.log_history)
    completed_steps = int(trainer.state.global_step)
    trainer.save_model(str(trainer_output / "final_adapter"))
    tokenizer.save_pretrained(str(trainer_output / "final_adapter"))

    checkpoints = sorted(trainer_output.glob("checkpoint-*"))
    reload_path = checkpoints[-1] if checkpoints else trainer_output / "final_adapter"
    del trainer
    del model
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    base_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    reloaded = PeftModel.from_pretrained(base_model, str(reload_path))
    reloaded.eval()
    reload_messages = [
        {
            "role": "user",
            "content": "Compute 2 + 3. Finish with exactly <answer>INTEGER</answer>.",
        }
    ]
    rendered = tokenizer.apply_chat_template(
        reload_messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = reloaded.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
        )
    reload_completion = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    torch.cuda.synchronize()
    reload_peak_allocated_bytes = torch.cuda.max_memory_allocated()
    reload_peak_reserved_bytes = torch.cuda.max_memory_reserved()

    training_loss = float(train_result.training_loss)
    finite_loss = math.isfinite(training_loss)
    reward_std_values = [
        float(item["reward_std"])
        for item in metric_records
        if "reward_std" in item and math.isfinite(float(item["reward_std"]))
    ]
    group_variation_observed = any(value > 0.0 for value in reward_std_values)
    checkpoint_written = bool(checkpoints) and reload_path.exists()
    adapter_changed = bool(changed_tensors)
    reload_generated = bool(reload_completion.strip())
    correct_rollouts = sum(bool(item["correct"]) for item in ROLLOUTS)
    format_valid_rollouts = sum(bool(item["format_valid"]) for item in ROLLOUTS)
    passed = (
        finite_loss
        and group_variation_observed
        and adapter_changed
        and checkpoint_written
        and reload_generated
        and completed_steps == MAX_STEPS
        and len(ROLLOUTS) == GROUP_SIZE * MAX_STEPS
    )
    python_elapsed_seconds = time.perf_counter() - program_started

    summary = {
        "adapter_changed": adapter_changed,
        "changed_trainable_tensors": len(changed_tensors),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_written": checkpoint_written,
        "completed_steps": completed_steps,
        "correct_rollouts": correct_rollouts,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "max_steps": MAX_STEPS,
        "python_elapsed_seconds": python_elapsed_seconds,
        "training_elapsed_seconds": training_elapsed_seconds,
        "final_trainable_sha256": final_digest,
        "finite_training_loss": finite_loss,
        "format_valid_rollouts": format_valid_rollouts,
        "group_variation_observed": group_variation_observed,
        "initial_trainable_sha256": initial_digest,
        "max_memory_allocated_bytes": max(
            training_peak_allocated_bytes, reload_peak_allocated_bytes
        ),
        "max_memory_reserved_bytes": max(
            training_peak_reserved_bytes, reload_peak_reserved_bytes
        ),
        "reload_peak_memory_allocated_bytes": reload_peak_allocated_bytes,
        "reload_peak_memory_reserved_bytes": reload_peak_reserved_bytes,
        "training_peak_memory_allocated_bytes": training_peak_allocated_bytes,
        "training_peak_memory_reserved_bytes": training_peak_reserved_bytes,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": resolved_revision,
        "model_snapshot_path": str(snapshot_path),
        "passed": passed,
        "platform": platform.platform(),
        "python": sys.version,
        "reload_generated": reload_generated,
        "reload_checkpoint_path": str(reload_path.resolve()),
        "rollout_count": len(ROLLOUTS),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "training_loss": training_loss,
        "versions": {
            "accelerate": accelerate.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "metrics.json", {"train": train_result.metrics, "log_history": metric_records})
    write_json(
        output_dir / "samples.json",
        {"reload_completion": reload_completion, "rollouts": ROLLOUTS},
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        print("GPU GRPO smoke test failed one or more pass criteria.", file=sys.stderr)
        return 1
    print("GPU GRPO smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
