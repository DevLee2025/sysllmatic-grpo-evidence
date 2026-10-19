# Scholar A40 GRPO infrastructure check

**Status: PASS.** This is a training-infrastructure check, not evidence that GRPO improves code optimization.

## Result

- Five-step run: 5/5 optimizer steps completed
- Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`
- GPU: NVIDIA A40
- Peak allocated GPU memory: 1.89 GiB (training peak: 1.27 GiB)
- Wall time: 228.00 s end to end; training 16.16 s; Python run 22.36 s
- Reloaded checkpoint: `/scratch/scholar/lee5897/sysllmatic-grpo-smoke/runs/gpu-20261019T053121Z-478360/trainer_output/checkpoint-5`

## Environment

- Slurm: job `478360`, node `scholar-j000`, partition `gpu`
- Python module: `anaconda/2024.10-py312`
- Python: `3.12.15 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:20:25) [GCC 14.3.0]`
- PyTorch/CUDA: `2.7.1+cu128` / `12.8`
- Transformers/TRL: `5.18.0` / `1.14.1`
- When produced, full records include `summary.json`, `config.json`, `metrics.json`, `samples.json`, `environment.txt`, `system-environment.txt`, `nvidia-smi.txt`, `source_manifest.sha256`, and logs in this directory.
