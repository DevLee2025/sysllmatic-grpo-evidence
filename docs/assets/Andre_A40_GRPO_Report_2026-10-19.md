# Scholar A40 GRPO progress report

On October 19, 2026, the prepared five-step GRPO smoke test completed successfully on one NVIDIA A40 through Purdue Scholar (Slurm job `478360`). The run used `Qwen/Qwen2.5-0.5B-Instruct` at immutable revision `7ae557604adf67be50417f59c2c2f167def9a775` with TRL 1.14.1, PyTorch 2.7.1, and CUDA 12.8.

## Result

- Completed all 5 optimizer steps and 20 rollouts.
- Changed 96 trainable LoRA tensors.
- Saved checkpoint 5, reloaded it, and generated a nonempty completion.
- Used 1.27 GiB peak allocated GPU memory during training and 1.89 GiB across the full Python run.
- Took 16.16 seconds for training, 22.36 seconds in Python, and 3 minutes 56 seconds of Slurm elapsed time including environment setup.
- Finished with exit code 0 on `scholar-j000` in the `gpu` partition.

## Interpretation

This establishes that the pinned CUDA environment, generation, GRPO update, LoRA checkpoint, reload, and artifact-export path work on Scholar. The learning signal was sparse: only 1 of 20 rollouts was correctly formatted and correct. The first step had reward standard deviation `0.5609` and gradient norm `1.4838`; steps 2–5 had zero within-group reward variance and zero gradient norm. The reload completion was nonempty but did not answer its arithmetic prompt correctly.

The run is therefore a successful infrastructure check, not evidence that five-step GRPO improved reasoning or code optimization. The next model-side deliverable is a fixed candidate generated from the saved adapter, paired with the base-checkpoint candidate under identical settings. Evaluator conclusions still require independent compilation and correctness checks before timing.

## Evidence

The complete audit-sized export is in [`experiments/grpo_smoke/results/gpu-20261019T053121Z-478360/`](../../experiments/grpo_smoke/results/gpu-20261019T053121Z-478360/). Its `summary.json`, `metrics.json`, `samples.json`, Slurm accounting, package freeze, source hashes, and logs support the figures above. Model weights and optimizer state remain in Scholar scratch; `checkpoint_manifest.sha256` records their hashes.
