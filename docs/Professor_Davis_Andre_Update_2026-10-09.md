# Andre update for Professor Davis — October 9, 2026

## Copy/paste response

**CPU GRPO pilot:** I implemented and ran a one-step, end-to-end GRPO plumbing test under WSL on CPU. It loads a pinned tiny Qwen checkpoint, generates a four-completion sibling group, evaluates rewards, backpropagates through a LoRA adapter, updates the adapter, saves and reloads the checkpoint, and generates again after reload. The run passed: the gradient norm was `0.00907`, the trainable-parameter hash changed, the checkpoint reloaded successfully, and the measured training portion took `1.46 s`. The source, pinned dependencies, exact configuration, metrics, samples, environment freeze, logs, and checkpoint hashes are included in the attached bundle.

**What I mean by “marginal”:** I do not mean that the pilot measured a small optimization improvement. It measured no program speedup and provides no evidence that GRPO improved model quality. The tiny model gave the same malformed-task reward to all four completions, so I added a fixed diagnostic probe reward to create within-group variation and verify that the optimizer path could actually update the adapter. “Marginal” therefore refers to the pilot's limited evidentiary value for the research hypothesis: it validates infrastructure, while contributing essentially no evidence yet about code-optimization performance. I will use “infrastructure-only result” instead because it is more precise.

**A40 check:** We confirmed access to a Scholar A40 allocation (`46,068 MiB`) and corrected the Scholar request to use account `scholar`, partition `gpu`, the `a40` constraint, and the partition's default QoS. I prepared a one-hour batch job for exactly five GRPO optimizer steps with `Qwen/Qwen2.5-0.5B-Instruct` pinned to commit `7ae557604adf67be50417f59c2c2f167def9a775`. It records peak allocated/reserved GPU memory, training and end-to-end time, package versions, source hashes, model revision, checkpoints, logs, and finalized Slurm accounting. The live five-step run has not completed; cluster authentication/execution remains the immediate blocker.

**Candidate generation:** I fixed the candidate generator to the same immutable Qwen checkpoint with greedy decoding, seed `42`, one returned sequence, and at most `2,048` new tokens. Every attempt stores the exact rendered prompt, raw completion, extracted source, hashes, model settings, token counts, elapsed time, environment, and empty evaluator fields for compilation, correctness, timing, and feedback. I generated one provisional candidate for ParallelKernelBench problem 1 (all-reduce) in `9.98 s` on CPU. The record passes schema validation, but the candidate reproduces the reference algorithm and has not been run through PKB, so it is an interface/handoff test rather than an optimization result. Once Manas confirms the selected task and evaluator command, I can generate the fixed starting candidate and hand over the unchanged attempt record for evaluation.

## Attachment contents

`deliverables/andre-grpo-pilot-2026-10-09.zip` contains:

- the CPU and Scholar training scripts;
- fixed-checkpoint candidate-generation code;
- pinned dependency files and usage instructions;
- the complete audit-sized CPU pilot result;
- the provisional candidate package and attempt schema; and
- the model/training-infrastructure status note.

No model weights, caches, credentials, or private keys are included.
