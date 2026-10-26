# Manas evaluator handoff

This page identifies the exact artifacts for the first model-to-evaluator integration and defines the pass condition. The target is one ParallelKernelBench reference task and one recorded candidate, with correctness established before any timing result is accepted.

## Direct artifacts

- [CPU pilot code](../../experiments/grpo_smoke/cpu_preflight.py) and [local runner](../../experiments/grpo_smoke/run_cpu_preflight.sh)
- [CPU pilot report](../../experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/REPORT.md), [metrics](../../experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/metrics.json), and [complete audit-sized output](../../experiments/grpo_smoke/results/20261004T205029Z_cpu_preflight/)
- [Scholar interactive runner](../../experiments/grpo_smoke/run_scholar_gpu.sh) and [Slurm batch job](../../experiments/grpo_smoke/scholar_gpu.sub)
- [Provisional PKB problem 1 task contract](../../experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/task.json)
- [Generated candidate](../../experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/candidate.py), [attempt record](../../experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/attempt.json), and [generation report](../../experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/REPORT.md)

The expected SHA-256 for `candidate.py` is:

```text
45c57a796706255764bd120f804263353e79c7d9c9d380204276d437efb7c701
```

The candidate clones the input and calls `torch.distributed.all_reduce` with `SUM`, exactly as the reference does. This makes it useful for proving the evaluator and record interface, but it is not an optimization result.

## Running the Scholar pilot interactively

`run_scholar_gpu.sh` can run directly from an authenticated interactive Scholar allocation. It does not require `sbatch`. Run it on the allocated GPU compute node, not the login node:

```bash
cd "$RCAC_SCRATCH"
git clone --branch main --single-branch \
  https://github.com/DevLee2025/sysllmatic-grpo-evidence.git
cd sysllmatic-grpo-evidence/experiments/grpo_smoke

test -n "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash run_scholar_gpu.sh
```

The script requires:

- an active CUDA GPU allocation and a populated `SLURM_JOB_ID`;
- `RCAC_SCRATCH` to be set and writable;
- either `anaconda/2024.10-py312`, the legacy fallback module, or an available Python 3.12 module supplied through `GRPO_PYTHON_MODULE`;
- access to the public PyTorch package index and Hugging Face model files on a cold first run, or previously populated caches.

The earlier blocker was the lack of a reusable authenticated cluster session for remote execution. There is no batch-only requirement in the runner. If the interactive command fails, the failure should identify a concrete environment dependency such as the module name, package/model download access, CUDA availability, scratch access, or wall-time expiration.

## Correctness-first evaluator sequence

1. Use the reference identified by `task.json`: `togethercomputer/ParallelKernelBench_Problems` revision `3ea6e263b29754fb639df0e25530c387f732fbcb`, path `reference/1_allreduce.py`.
2. Copy the recorded candidate and verify its SHA-256 before execution. Preserve the candidate source unchanged during evaluation.
3. Run the reference and candidate under the same world size, tensor shapes, dtypes, seeds, and hardware. Use rank-varying inputs so a missing or incorrect collective cannot pass as a local copy.
4. On every rank, compare the candidate output with the reference using the benchmark's tolerance. Also verify that the input tensor is unchanged and the output is a distinct tensor.
5. Record compilation/import, execution, and correctness status, plus worker logs and timeout cleanup, in a copy of `attempt.json`.
6. Only after all correctness cases pass, run the reference and candidate timing paths with identical warmup, synchronization, and measured iterations. Do not report a speedup for an incorrect or incomplete result.

## Confirmed first integration

Manas selected PKB problem `1_allreduce` at evaluator revision
`3cc0f9b435010b4957ccf7b4b85791480c7c69a6`, backend `cuda`, on two Scholar
A40 ranks. The replacement task contract is
[`pkb1_allreduce_scholar_a40_task.json`](../../experiments/grpo_smoke/pkb1_allreduce_scholar_a40_task.json).
It generates the exact filename expected by PKB, `1_allreduce_cuda.py`.

After receiving the checkpoint-5 candidate package, evaluate it with:

```bash
uv run python run_local.py \
  --nproc-per-node 2 \
  --mode eval \
  --problem 1 \
  --solution cuda \
  --solutions-dir "$(pwd)/solutions_andre_adapter" \
  --dtype bfloat16 \
  --trials 5 \
  --atol 1e-2 \
  --rtol 1e-2 \
  --download
```

This command tests correctness only. Add performance measurement only after all
five trials pass, while preserving the same evaluator revision and hardware.

The generated adapter package is
[`results/pkb1-adapter-480810/`](../../experiments/grpo_smoke/results/pkb1-adapter-480810/).
Use attempt ID `69d2f9e944314603` and verify candidate SHA-256
`45c57a796706255764bd120f804263353e79c7d9c9d380204276d437efb7c701`
before execution. The package already contains the required filename
`1_allreduce_cuda.py`; point `--solutions-dir` directly at the package directory
or copy the complete directory to `solutions_andre_adapter`.
