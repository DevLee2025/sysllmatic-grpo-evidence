# GRPO smoke test

This directory contains a deliberately small GRPO training smoke test. Its
purpose is to verify the training plumbing before allocating GPU time to a
code-optimization experiment. It does **not** measure whether GRPO improves
optimization performance.

The CPU preflight checks that the following operations complete together:

- load a public causal-language-model checkpoint;
- generate four completions for one prompt group;
- call a deterministic task reward and a diagnostic probe reward;
- compute a GRPO loss and backpropagate it;
- update at least one LoRA parameter;
- save and reload a checkpoint; and
- generate from the reloaded checkpoint.

## Local WSL run

Run from Ubuntu under WSL 2:

```bash
bash /mnt/c/path/to/SysLLMaticFall2026/experiments/grpo_smoke/run_cpu_preflight.sh
```

The wrapper creates an isolated environment at
`~/code/sysllmatic-grpo-smoke/.venv`, keeps Hugging Face and pip caches under
that work directory, and writes run artifacts beneath
`~/code/sysllmatic-grpo-smoke/runs/`. Override the location with
`GRPO_SMOKE_WORK_ROOT` if needed.

The generated checkpoint and package caches should not be committed to Git.
After a run, use `export_run_artifacts.py` to copy the audit-sized result into a
dated directory under `results/`. It copies the core JSON, environment, and log
files, includes GPU/Slurm records when present, and records hashes for the
retained checkpoint without adding model weights to Git. New exports use the
platform-neutral `artifact_size.txt`; the legacy `wsl_artifact_size.txt` alias
is also written so existing CPU automation continues to work.

## Pass criteria

The script exits successfully only when:

1. the single optimizer step completes with finite training loss;
2. at least one trainable LoRA tensor changes;
3. a checkpoint is written and reloads into the base model; and
4. the reloaded model produces a completion.

The tiny random checkpoint is not expected to solve the arithmetic prompt, so
the preflight adds a synthetic `probe_reward` with four fixed values. This
guarantees within-group variation and makes a policy-gradient update testable.
The arithmetic reward is unit-tested and logged separately. The probe is a
plumbing diagnostic and must not be used in the GPU experiment or interpreted
as learning evidence.

## GPU follow-up

The GPU smoke test uses `Qwen/Qwen2.5-0.5B-Instruct` pinned to immutable Hub
revision `7ae557604adf67be50417f59c2c2f167def9a775`, LoRA, five optimizer steps,
four generations per prompt, and one CUDA GPU. On Purdue Scholar, copy this
directory to scratch and run it from an active A40 allocation:

```bash
bash run_scholar_gpu.sh
```

For a fresh batch allocation, submit from this directory:

```bash
JOB_ID=$(sbatch --parsable scholar_gpu.sub)
echo "$JOB_ID"
```

The batch request allows one hour so a cold cache has time to install the CUDA
wheel and download the pinned model. Reused environments and caches should make
later runs substantially shorter.

The runner prefers Purdue's current `anaconda/2024.10-py312` module and retains
a legacy module-name fallback. If Scholar exposes a different Python 3.12
module, set it explicitly, for example:

```bash
GRPO_PYTHON_MODULE=anaconda/<available-py312-version> bash run_scholar_gpu.sh
```

`SCHOLAR_CONDA_MODULE` is accepted as a compatibility alias for the same
override. `run_scholar_candidate.sh` uses the identical module selection.

It creates an isolated prefix under
`$RCAC_SCRATCH/sysllmatic-grpo-smoke`, installs the pinned CUDA stack, and
retains caches and run artifacts in scratch. It uses only measured arithmetic
correctness, output format, and correct-answer brevity rewards; the CPU
preflight's synthetic probe reward is not present.

Every invocation writes a `REPORT.md`, including failed setup or training runs
when the system Python remains available. The report records completion status,
optimizer steps, peak allocated GPU memory, training and end-to-end wall time,
the reloaded checkpoint path, the exact model revision, and the main environment
versions. Supporting files include `summary.json`, the complete `pip freeze`,
`nvidia-smi`, Slurm metadata, logs, and source hashes.

After a batch job finishes, capture finalized Slurm accounting from the login
node. The in-job snapshot may still show the batch step as running:

```bash
RUN_DIR=$(ls -dt "$RCAC_SCRATCH"/sysllmatic-grpo-smoke/runs/gpu-*-"$JOB_ID" | head -n 1)
sacct -j "$JOB_ID" \
  --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode \
  > "$RUN_DIR/slurm-sacct-final.txt"
```

Keep the first successful run's `environment.txt` with the results. Reuse that
environment for the controlled comparison; if the environment must be rebuilt,
install from that exact frozen package set rather than resolving new transitive
versions.

To retain the latest run without copying its model weights:

```bash
RUN_DIR=${RUN_DIR:-$(ls -dt "$RCAC_SCRATCH"/sysllmatic-grpo-smoke/runs/gpu-* | head -n 1)}
python export_run_artifacts.py \
  --run-dir "$RUN_DIR" \
  --destination "results/$(basename "$RUN_DIR")"
```

A passing run confirms that the five-step training, checkpoint, and reload path
works on the selected GPU. It is not evidence that GRPO improves program
optimization.

## One-candidate interface for the evaluator

`generate_candidate.py` gives Manas's evaluator one source file plus a shared
attempt record. Both training and generation use
`Qwen/Qwen2.5-0.5B-Instruct` at immutable Hugging Face commit
`7ae557604adf67be50417f59c2c2f167def9a775`. Generation is greedy, uses seed
42, requests one sequence, and allows at most 2,048 new tokens. The script
records the rendered prompt, resolved snapshot, adapter manifest (when used),
package versions, token counts, and elapsed generation time.

First, copy `candidate_task.example.json` and replace every placeholder with
the selected PKB task, complete reference implementation, confirmed hardware,
and exact entrypoint. Validate the task contract locally without downloading a
model:

```bash
python experiments/grpo_smoke/generate_candidate.py \
  --task /path/to/pkb_task.json \
  --validate-only
```

From a Scholar GPU allocation, generate one base-checkpoint candidate with the
same environment and cache as the five-step smoke test:

```bash
bash experiments/grpo_smoke/run_scholar_candidate.sh \
  --task /path/to/pkb_task.json \
  --output-dir "$RCAC_SCRATCH/sysllmatic-grpo-smoke/candidates/pkb-task-base"
```

To generate from the five-step LoRA checkpoint instead, add its saved adapter:

```bash
bash experiments/grpo_smoke/run_scholar_candidate.sh \
  --task /path/to/pkb_task.json \
  --adapter-path "$RCAC_SCRATCH/sysllmatic-grpo-smoke/runs/gpu-RUN_ID/trainer_output/final_adapter" \
  --output-dir "$RCAC_SCRATCH/sysllmatic-grpo-smoke/candidates/pkb-task-five-step"
```

Each output directory is self-contained: `task.json`, `rendered_prompt.txt`,
`raw_completion.txt`, `candidate.py` (or the language-appropriate extension),
and `attempt.json`. The attempt record follows `attempt_record.schema.json`.
Its compilation, execution, correctness, timing, and feedback fields start as
`not_run`; the evaluator fills them after checking the unchanged candidate
hash. The five-step adapter is an infrastructure artifact, not evidence that
the resulting candidate is faster or more correct. The Scholar wrapper also
passes `--require-cuda`, so it fails instead of silently running on a login
node without a GPU. Direct invocation may fall back to CPU, and the attempt
record always captures the actual device and dtype.

A complete CPU interface check is archived under
`results/20261005T211133Z_pkb1_candidate_interface_cpu/`. It generated one
candidate for a provisional, revision-pinned PKB problem 1 contract and wrote a
valid attempt record. The candidate uses the same algorithm as the reference;
the archive proves the handoff format, not correctness testing or optimization.
