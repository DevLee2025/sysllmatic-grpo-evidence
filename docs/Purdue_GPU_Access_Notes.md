# Purdue GPU access for the GRPO smoke test

## Which system to use

Anvil and Scholar Spark are separate systems. Spark is a set of Arm-based GPU
nodes reached through Purdue Scholar; it is not part of Anvil.

For the five-step Qwen2.5-0.5B GRPO smoke test, the preferred order is:

1. Scholar's ordinary x86 `gpu` partition, if the VIP class has a Scholar
   account;
2. Anvil, if the team already has an NSF ACCESS or NAIRR allocation there; and
3. Scholar `spark-batch` only if an ordinary GPU is unavailable.

The toy run needs one GPU. It does not need an eight-GPU H100 node.

## Scholar ordinary GPU

The canonical hostname is `scholar.rcac.purdue.edu`. From PowerShell or WSL:

```bash
ssh <Purdue-Career-username>@scholar.rcac.purdue.edu
```

Scholar class access is created by the instructor for a semester and CRN. Once
logged in, inspect the system with:

```bash
slist
sfeatures
findscratch
echo "$RCAC_SCRATCH"
myquota
```

Scholar class jobs use account `scholar`. The ordinary GPU partition is `gpu`.
Scholar's GPU documentation does not require a user-specified QOS; the
partition applies its default policy. Request a 30-minute interactive job with:

```bash
sinteractive -A scholar --partition=gpu \
  --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
```

To target a full x86 A40 for the smoke test and avoid MIG or Arm nodes, append
`--mem=32G --constraint=a40`. If no A40 is available, use
`--constraint=a30`.

After allocation, record `nvidia-smi` and use
`$RCAC_SCRATCH/sysllmatic-grpo-smoke` for the environment, model cache, and
checkpoint. Scholar currently documents V100, A30, and A40-class nodes, any of
which is sufficient for this test.

Official references: [Scholar accounts](https://docs.rcac.purdue.edu/userguides/scholar/accounts/),
[queues and QOS](https://docs.rcac.purdue.edu/userguides/scholar/run_jobs/queues/),
[GPU jobs](https://docs.rcac.purdue.edu/userguides/scholar/run_jobs/gpu/), and
[hardware](https://docs.rcac.purdue.edu/userguides/scholar/overview/).

## Anvil

Anvil is an NSF ACCESS/NAIRR resource operated by Purdue. A Purdue account alone
does not grant Anvil compute time. A project manager must add the user to an
allocation, and ACCESS credits must be assigned to the relevant Anvil resource.

The canonical hostname is `anvil.rcac.purdue.edu`. Anvil usernames are commonly
prefixed with `x-`, and login is SSH-key based:

```bash
ssh x-<ACCESS-username>@anvil.rcac.purdue.edu
```

After logging in, discover the requested values with:

```bash
mybalance
showpartitions
sfeatures
echo "$SCRATCH"
myquota
```

- `mybalance` supplies the Slurm allocation/account to pass with `-A`.
- `showpartitions` lists the partitions available to that allocation.
- `gpu-debug` is the natural 30-minute test partition when the allocation can
  use it. Request one GPU and one node.
- Anvil's documented `gpu` nodes use A100s and its `ai` nodes use H100s.
- A user-provided QOS is ordinarily unnecessary on Anvil; follow the values
  shown for the allocation and partition.

Use `$SCRATCH/sysllmatic-grpo-smoke` for temporary files. Anvil scratch is
unbacked temporary storage and is subject to its purge policy.

Official references: [Anvil access](https://docs.rcac.purdue.edu/userguides/anvil/access/),
[getting started](https://docs.rcac.purdue.edu/userguides/anvil/getting-started/),
[jobs](https://docs.rcac.purdue.edu/userguides/anvil/jobs/),
[architecture](https://docs.rcac.purdue.edu/userguides/anvil/architecture/), and
[file management](https://docs.rcac.purdue.edu/userguides/anvil/file_management/).

## Scholar Spark

Spark uses the same Scholar hostname and account. Its GPU partitions are
`spark-interactive` and `spark-batch`. Each Spark node has one NVIDIA GB10 GPU
with unified memory, but the host architecture is Arm64. Existing x86 virtual
environments will not run there.

If Spark is the available route, begin with:

```bash
module load modtree/spark
```

Then build a fresh Arm-compatible environment and request one GPU in
`spark-batch`. This is a workable fallback, but it adds packaging risk that is
irrelevant to the GRPO question. See Purdue's [Scholar Spark guide](https://docs.rcac.purdue.edu/userguides/scholar/scholar-spark/).

## Finding or creating an SSH alias

An alias is optional; the canonical hostname is sufficient. To inspect only
the non-secret host fields in an existing Windows SSH configuration:

```powershell
Select-String -Path "$HOME\.ssh\config" -Pattern '^\s*(Host|HostName|User)\s+'
```

Resolve an existing alias without displaying private keys:

```powershell
ssh -G <alias> | Select-String '^(hostname|user) '
```
