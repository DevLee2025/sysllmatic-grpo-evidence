# Fixed-checkpoint candidate interface check

**Status: PASS.** The pinned model produced one self-contained candidate and a
schema-compatible attempt record. This validates the generation handoff only;
it is not a PKB correctness, timing, optimization, or GPU result.

## Result

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Task: provisional PKB problem 1 all-reduce contract, pinned to dataset
  revision `3ea6e263b29754fb639df0e25530c387f732fbcb`
- Generation: greedy, seed 42, one sequence, at most 2,048 new tokens
- Device: CPU (`torch 2.7.1+cpu`) under WSL 2
- Generation time: 9.98 seconds
- Input/output tokens: 316 / 56
- Attempt ID: `a5f3c96231e3041c`

The generated `candidate.py` uses the same operations and algorithm as the
supplied reference implementation. That makes it useful as a first evaluator
plumbing input, but it does not demonstrate an optimization. Manas should
replace the provisional task contract with the team-selected PKB task and
confirmed hardware before the controlled experiment.

## Files

- `task.json`: exact task and hardware assumptions supplied to the model
- `rendered_prompt.txt`: exact chat-template input
- `raw_completion.txt`: unmodified model response
- `candidate.py`: extracted evaluator input
- `attempt.json`: model, generation, environment, hashes, and empty evaluator
  result fields
