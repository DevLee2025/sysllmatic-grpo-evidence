# CPU GRPO preflight report

**Run:** `20261004T205029Z`
**Date:** October 4, 2026
**Result:** PASS
**Environment:** Ubuntu 24.04 under WSL 2, CPU only

## What passed

- TRL 1.14.1 loaded the public tiny Qwen2 test checkpoint at revision
  `ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`.
- Native Transformers generation produced one four-completion sibling group.
- Hand-written fixtures verified the arithmetic reward's correct, valid-wrong,
  and malformed cases as `1.0`, `0.0`, and `-1.0`.
- The diagnostic probe reward produced `reward_std = 0.8606629968` and
  `frac_reward_zero_std = 0.0`.
- The backward pass produced `grad_norm = 0.0090739308`.
- The trainable LoRA digest changed from
  `9cdab0597887e9464e36f1d01bc2014145f060d7234b07c52b8b65471a6f2b8c`
  to
  `c70aef19373267a787f29c17c23392abf9d22cddb0448240506d5feef6b234d9`.
- `checkpoint-1` was written, reloaded into the base checkpoint, and generated
  a completion.

The reported scalar loss is `0.0`. That is valid for this first on-policy,
group-centered step: the nonzero gradient norm and changed adapter digest are
the direct evidence that an optimizer update occurred.

## Diagnostic reward

The tiny checkpoint generated four malformed arithmetic responses, so the real
task reward was `-1.0` for every sibling. A task-only run would therefore have
zero advantages and could not test the optimizer path. This preflight adds the
fixed probe values `[-1, -1/3, 1/3, 1]` to guarantee group variation. The probe
is plumbing instrumentation. It must be removed from the GPU experiment and is
not evidence that GRPO learned arithmetic or code optimization.

## Reproducibility record

- Training script SHA-256:
  `4bd2318069ecd49208f180ca5258cb79d5c74691cae26b6edc90ceddbcf0139b`
- Training time reported by the script: `1.4632` seconds.
- Complete authorized WSL work area after installation and three diagnostic
  attempts: approximately `1.7G`.
- Final run directory, including the uncommitted checkpoint: `22,938,199`
  bytes.
- WSL checkpoint:
  `/home/andre/code/sysllmatic-grpo-smoke/runs/20261004T205029Z/trainer_output/checkpoint-1`

`checkpoint_manifest.sha256` records the checkpoint file hashes without
placing model weights or optimizer state in Git. `environment.txt` contains the
fully resolved package set. `config.json`, `metrics.json`, and `samples.json`
contain the exact configuration, trainer metrics, completions, token IDs, and
task rewards.

## Setup observations

The Ubuntu image did not contain `ensurepip`, so `python3 -m venv` initially
failed. The runner now bootstraps pinned `virtualenv==20.31.2` inside the
authorized experiment directory and requires no `sudo` or system-package
change.

An initial task-only diagnostic produced uniform rewards, zero gradient norm,
and no adapter change. This was the expected failure mode for a random tiny
model and led to the explicit probe-reward design used in the passing run.

## Scope

This result validates the local Python, Transformers, PEFT, TRL, grouped reward,
backpropagation, checkpoint, and reload plumbing. It does not validate CUDA,
NCCL, a useful model, code execution rewards, multi-turn credit assignment, or
multi-GPU kernel optimization.
