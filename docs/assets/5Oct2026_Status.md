# October 5, 2026 Project Status

## Deliverable status

| Deliverable | Status | Evidence |
|---|---|---|
| Five-step Qwen/A40 training check | Ready to run on Scholar; cluster authentication is the remaining execution dependency | `experiments/grpo_smoke/run_scholar_gpu.sh` and `scholar_gpu.sub` |
| Fixed model/checkpoint | Complete | `Qwen/Qwen2.5-0.5B-Instruct` at immutable Hub revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| Reproducible one-candidate interface | Complete | `experiments/grpo_smoke/generate_candidate.py` and `attempt_record.schema.json` |
| Recorded candidate | Complete as an interface check | `experiments/grpo_smoke/results/20261005T211133Z_pkb1_candidate_interface_cpu/` |

The five-step run is a training-infrastructure check. The recorded candidate is
a generation-interface check. Neither result measures whether GRPO improves
code optimization.

## Fixed generation settings

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Decoding: greedy (`do_sample=false`, one beam, one returned sequence)
- Seed: `42`
- Maximum new tokens: `2048`
- Prompt contract: one complete source file in one fenced code block, with the
  task reference, hardware, interface, and constraints embedded in the prompt

The output package contains the exact prompt, raw completion, extracted source,
hashes, package versions, token counts, elapsed generation time, and empty
fields for Manas's evaluator to fill with compilation, execution, correctness,
timing, and feedback results.

## Recorded interface check

The pinned base checkpoint generated one candidate for a provisional,
revision-pinned PKB problem 1 all-reduce contract. The run completed on CPU in
WSL 2 in 9.98 seconds and produced attempt ID `a5f3c96231e3041c`.
The candidate uses the same algorithm as the reference. This is useful for
testing the candidate/evaluator handoff, but it is not an optimization result.

The provisional task came from
`togethercomputer/ParallelKernelBench_Problems` revision
`3ea6e263b29754fb639df0e25530c387f732fbcb`. Manas should replace it with the
team-selected task and confirmed hardware before the controlled comparison.

## Scholar A40 run

From the directory containing the scripts, submit:

```bash
JOB_ID=$(sbatch --parsable scholar_gpu.sub)
echo "$JOB_ID"
```

The job requests one full Scholar A40 for one hour, creates or reuses an
isolated Python 3.12 environment in `$RCAC_SCRATCH`, downloads the pinned stack,
runs exactly five GRPO optimizer steps, saves and reloads the LoRA checkpoint,
and writes `REPORT.md` plus machine-readable logs. The report includes the
completion result, peak allocated GPU memory, training time, end-to-end time,
checkpoint path, package versions, source hashes, and Slurm metadata.

After the job finishes, capture finalized accounting before export:

```bash
RUN_DIR=$(ls -dt "$RCAC_SCRATCH"/sysllmatic-grpo-smoke/runs/gpu-*-"$JOB_ID" | head -n 1)
sacct -j "$JOB_ID" \
  --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode \
  > "$RUN_DIR/slurm-sacct-final.txt"
```

The archived `environment.txt` is the exact package freeze for subsequent
controlled comparisons.

After the run, retain only the audit-sized artifacts:

```bash
RUN_DIR=${RUN_DIR:-$(ls -dt "$RCAC_SCRATCH"/sysllmatic-grpo-smoke/runs/gpu-* | head -n 1)}
python export_run_artifacts.py \
  --run-dir "$RUN_DIR" \
  --destination "results/$(basename "$RUN_DIR")"
```

Model weights and caches remain in scratch. The repository receives the report,
JSON metrics, completions, environment freeze, GPU and Slurm records, logs, and
a checkpoint hash manifest.

## Evaluator handoff

Once Manas supplies the selected task contract, generate the frozen base-model
candidate from a Scholar GPU allocation:

```bash
bash run_scholar_candidate.sh \
  --task /path/to/selected-pkb-task.json \
  --output-dir "$RCAC_SCRATCH/sysllmatic-grpo-smoke/candidates/selected-task-base"
```

The evaluator should preserve the candidate hash and fill the existing
`evaluation` fields rather than changing the generated source or generation
metadata.
