# SysLLMatic GRPO evidence

This repository is a public evidence snapshot for Andre Lee's Fall 2026 work in Purdue University's Vertically Integrated Projects Pre-trained Models team. It documents a small GRPO training-infrastructure pilot, a fixed candidate-generation interface, and the completed Purdue Scholar A40 follow-up.

The completed result is intentionally narrow: the CPU pilot verifies the basic path, and the A40 run verifies five optimizer steps with a pinned Qwen checkpoint and real arithmetic reward. Neither shows that GRPO improves code correctness or speed. The archived candidate demonstrates the model-to-evaluator handoff format and has not been evaluated as an optimization.

## Evidence map

| Claim | Primary evidence |
|---|---|
| The one-step CPU GRPO path completed and changed a trainable LoRA tensor | [`REPORT.md`](experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/REPORT.md), [`metrics.json`](experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/metrics.json), and [`run.log`](experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/run.log) |
| The saved checkpoint was hashed and reloaded | [`checkpoint_manifest.sha256`](experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/checkpoint_manifest.sha256) and [`summary.json`](experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/summary.json) |
| The candidate generator records fixed settings and a shared attempt schema | [`generate_candidate.py`](experiments/grpo_smoke/generate_candidate.py), [`attempt_record.schema.json`](experiments/grpo_smoke/attempt_record.schema.json), and the [archived attempt](experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/attempt.json) |
| The five-step Qwen GPU path completed on a Scholar A40 | [concise report](docs/assets/Andre_A40_GRPO_Report_2026-10-19.md), [run report](experiments/grpo_smoke/results/gpu-20261019T053121Z-478360/REPORT.md), and [machine-readable summary](experiments/grpo_smoke/results/gpu-20261019T053121Z-478360/summary.json) |
| The earlier model-interface status and limitations are documented | [October 5 status](docs/assets/5Oct2026_Status.md) and [Manas evaluator handoff](docs/assets/Manas_Evaluator_Handoff.md) |

For the first evaluator integration, use the [Manas evaluator handoff](docs/assets/Manas_Evaluator_Handoff.md). It links the exact task, candidate, run artifacts, and Scholar scripts and defines the correctness gate required before timing.

The complete operating instructions and pass criteria are in [`experiments/grpo_smoke/README.md`](experiments/grpo_smoke/README.md).

## Reproduce the CPU preflight

From Ubuntu under WSL 2:

```bash
bash /mnt/c/path/to/sysllmatic-grpo-evidence/experiments/grpo_smoke/run_cpu_preflight.sh
```

The wrapper creates an isolated environment and keeps downloaded models, package caches, and generated checkpoints outside the repository. Exact package pins are in [`requirements-cpu.txt`](experiments/grpo_smoke/requirements-cpu.txt). The committed result is audit-sized and excludes model weights.

## Fixed candidate settings

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Decoding: greedy, one sequence
- Seed: `42`
- Maximum new tokens: `2048`

The archived ParallelKernelBench problem 1 candidate reproduces the reference algorithm. It validates the record format only; compilation, correctness, and performance remain marked `not_run`.

## Scope and provenance

This is a student-maintained evidence repository prepared for course and project review. It is not an official Purdue University or Purdue Duality Lab release. The original project repository and unrelated team work are intentionally excluded. No credentials, private keys, model weights, package caches, or unpublished evaluator data are included.

No repository-wide reuse license is asserted in this snapshot. Referenced models, datasets, repositories, and software remain subject to their respective licenses.
