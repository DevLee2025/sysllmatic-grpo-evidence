# PKB problem 1 checkpoint-5 candidate

**Status: GENERATED; evaluation pending.** Scholar job `480810` completed with
exit code 0 and produced one fixed candidate from the five-step LoRA checkpoint.
Compilation, execution, correctness, and timing have not yet been run.

## Identity

- Attempt ID: `69d2f9e944314603`
- Candidate: `1_allreduce_cuda.py`
- Candidate SHA-256: `45c57a796706255764bd120f804263353e79c7d9c9d380204276d437efb7c701`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Adapter manifest SHA-256: `0737a0b68b878eb0d271b3d0ae29efd5d0dedc6402979fb145be18f180ed8399`
- Task SHA-256: `48b4ae4152456e64179d54106c1a0f7e7068e78407aef37012739448acbc3821`
- Prompt SHA-256: `dd7c73f53fe93be4a311304e9bbc86ac0d4798bf1ec4b7f6e4805f3b453848de`
- Generation: greedy, seed 42, one sequence, 9.23 seconds on an NVIDIA A40

## Evaluator contract

- ParallelKernelBench revision: `3cc0f9b435010b4957ccf7b4b85791480c7c69a6`
- Problem/backend: `1_allreduce` / `cuda`
- Hardware: two Scholar A40 ranks
- Correctness: five bfloat16 trials at `atol=1e-2`, `rtol=1e-2`

From the pinned PKB checkout, point `--solutions-dir` at this package:

```bash
uv run python run_local.py \
  --nproc-per-node 2 \
  --mode eval \
  --problem 1 \
  --solution cuda \
  --solutions-dir /path/to/pkb1-adapter-480810 \
  --dtype bfloat16 \
  --trials 5 \
  --atol 1e-2 \
  --rtol 1e-2 \
  --download
```

The generated source reproduces the reference PyTorch/NCCL algorithm, so this
is an evaluator-plumbing candidate rather than an optimized kernel. Its source
hash equals the earlier provisional base candidate's hash. Because that earlier
generation used a different task contract, a new base-checkpoint generation
with this confirmed contract is required for a controlled adapter comparison.

## Files

- `attempt.json`: complete model, adapter, task, generation, and empty evaluation metadata
- `task.json`: exact confirmed task contract
- `rendered_prompt.txt` and `raw_completion.txt`: exact model input and output
- `generation_manifest.sha256`: hashes recorded on Scholar
- `scholar-job-480810.log`: scheduler, generation, hash-check, and metadata output
