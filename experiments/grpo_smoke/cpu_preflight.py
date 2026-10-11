#!/usr/bin/env python3
"""One-step CPU GRPO preflight with explicit, machine-readable checks."""

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
import torch
import transformers
import trl
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
GROUP_SIZE = 4
PROBE_PATTERN = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
ANSWER_PATTERN = re.compile(r"^\s*<answer>\s*(-?\d+)\s*</answer>\s*$", re.IGNORECASE)
ROLLOUTS: list[dict[str, Any]] = []


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trainable_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_answer(completion: str, expected: str) -> float:
    match = ANSWER_PATTERN.fullmatch(completion)
    if match is None:
        return -1.0
    return 1.0 if match.group(1) == str(expected) else 0.0


def arithmetic_reward(
    completions: list[str],
    answer: list[str],
    completion_ids: list[list[int]] | None = None,
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    ids = completion_ids or [[] for _ in completions]
    for completion, token_ids, expected in zip(completions, ids, answer, strict=True):
        reward = score_answer(completion, expected)
        rewards.append(reward)
        ROLLOUTS.append(
            {
                "answer": expected,
                "completion": completion,
                "completion_ids": list(token_ids),
                "task_reward": reward,
            }
        )
    return rewards


def probe_reward(completions: list[str], **_: Any) -> list[float]:
    """Force within-group variation solely to test the GRPO update path."""
    if len(completions) % GROUP_SIZE != 0:
        raise ValueError("probe reward received an incomplete generation group")
    return [PROBE_PATTERN[index % GROUP_SIZE] for index in range(len(completions))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    trainer_output = output_dir / "trainer_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    fixture_scores = [
        score_answer("<answer>5</answer>", "5"),
        score_answer("<answer>6</answer>", "5"),
        score_answer("5", "5"),
    ]
    if fixture_scores != [1.0, 0.0, -1.0]:
        raise AssertionError(f"reward fixture failed: {fixture_scores}")
    if trl.__version__ != "1.14.1":
        raise AssertionError(f"expected TRL 1.14.1, found {trl.__version__}")

    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Compute 2 + 3. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 7 - 4. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 3 * 4. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 8 + 6. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 9 - 2. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 5 * 2. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 4 + 4. Reply exactly as <answer>INTEGER</answer>.",
                "Compute 6 - 1. Reply exactly as <answer>INTEGER</answer>.",
            ],
            "answer": ["5", "3", "12", "14", "7", "10", "8", "5"],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    snapshot_path = Path(snapshot_download(repo_id=MODEL_ID, local_files_only=True))
    resolved_model_revision = snapshot_path.name if snapshot_path.parent.name == "snapshots" else None
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    config = GRPOConfig(
        output_dir=str(trainer_output),
        max_steps=1,
        per_device_train_batch_size=GROUP_SIZE,
        gradient_accumulation_steps=1,
        num_generations=GROUP_SIZE,
        max_completion_length=16,
        learning_rate=0.1,
        beta=0.0,
        loss_type="dr_grpo",
        scale_rewards="none",
        reward_weights=[1.0, 1.0],
        temperature=1.0,
        use_cpu=True,
        use_vllm=False,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=1,
        logging_strategy="steps",
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_steps=1,
        save_strategy="steps",
        save_total_limit=1,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[arithmetic_reward, probe_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    initial_digest = trainable_digest(trainer.model)
    started = time.time()
    train_result = trainer.train()
    elapsed_seconds = time.time() - started
    final_digest = trainable_digest(trainer.model)
    trainer.save_model(str(trainer_output / "final_adapter"))
    tokenizer.save_pretrained(str(trainer_output / "final_adapter"))

    checkpoints = sorted(trainer_output.glob("checkpoint-*"))
    reload_path = checkpoints[-1] if checkpoints else trainer_output / "final_adapter"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base_model, str(reload_path))
    reloaded.eval()
    encoded = tokenizer("Compute 2 + 3. Finish with <answer>", return_tensors="pt")
    with torch.no_grad():
        generated = reloaded.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(generated[0], skip_special_tokens=True)

    metrics = dict(train_result.metrics)
    metric_records = list(trainer.state.log_history)
    training_loss = float(train_result.training_loss)
    finite_loss = math.isfinite(training_loss)
    reward_std_values = [
        float(item["reward_std"])
        for item in metric_records
        if "reward_std" in item and math.isfinite(float(item["reward_std"]))
    ]
    zero_std_fractions = [
        float(item["frac_reward_zero_std"])
        for item in metric_records
        if "frac_reward_zero_std" in item
    ]
    group_variation_observed = any(value > 0 for value in reward_std_values) and any(
        value < 1.0 for value in zero_std_fractions
    )
    adapter_changed = initial_digest != final_digest
    checkpoint_written = bool(checkpoints) and reload_path.exists()
    reload_generated = bool(completion.strip())

    passed = (
        finite_loss
        and group_variation_observed
        and adapter_changed
        and checkpoint_written
        and reload_generated
        and len(ROLLOUTS) == GROUP_SIZE
    )
    summary = {
        "adapter_changed": adapter_changed,
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_written": checkpoint_written,
        "elapsed_seconds": elapsed_seconds,
        "final_trainable_sha256": final_digest,
        "finite_training_loss": finite_loss,
        "group_variation_observed": group_variation_observed,
        "initial_trainable_sha256": initial_digest,
        "model_id": MODEL_ID,
        "model_revision": resolved_model_revision or getattr(model.config, "_commit_hash", None),
        "model_snapshot_path": str(snapshot_path),
        "passed": passed,
        "platform": platform.platform(),
        "python": sys.version,
        "reload_generated": reload_generated,
        "reward_fixture_scores": fixture_scores,
        "rollout_count": len(ROLLOUTS),
        "torch": torch.__version__,
        "training_loss": training_loss,
        "trl_output_dir": str(trainer_output),
        "versions": {
            "accelerate": accelerate.__version__,
            "datasets": datasets.__version__,
            "peft": __import__("peft").__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
    }
    write_json(output_dir / "config.json", config.to_dict())
    write_json(output_dir / "metrics.json", {"train": metrics, "log_history": metric_records})
    write_json(
        output_dir / "samples.json",
        {"reload_prompt_and_completion": completion, "rollouts": ROLLOUTS},
    )
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        print("CPU GRPO preflight failed one or more pass criteria.", file=sys.stderr)
        return 1
    print("CPU GRPO preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
