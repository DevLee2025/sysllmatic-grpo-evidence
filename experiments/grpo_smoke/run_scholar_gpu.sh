#!/usr/bin/env bash
set -Eeuo pipefail

RUN_STARTED_EPOCH="$(date +%s)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${RCAC_SCRATCH:?RCAC_SCRATCH is not set; run this script on Scholar}"
WORK_ROOT="${GRPO_SMOKE_WORK_ROOT:-${RCAC_SCRATCH}/sysllmatic-grpo-smoke}"
ENV_DIR="${WORK_ROOT}/envs/trl-py312"
CACHE_DIR="${WORK_ROOT}/cache"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}"
RUN_DIR="${WORK_ROOT}/runs/gpu-${RUN_ID}"

mkdir -p \
  "${WORK_ROOT}/envs" \
  "${CACHE_DIR}/huggingface" \
  "${CACHE_DIR}/pip" \
  "${WORK_ROOT}/tmp" \
  "${RUN_DIR}"

touch "${RUN_DIR}/runner.log"
exec > >(tee -a "${RUN_DIR}/runner.log") 2>&1

REPORT_PYTHON="$(command -v python3 || command -v python || true)"
LOADED_PYTHON_MODULE="not loaded"

finalize() {
  local status=$?
  local ended_epoch elapsed_seconds
  trap - EXIT
  set +e
  ended_epoch="$(date +%s)"
  elapsed_seconds="$((ended_epoch - RUN_STARTED_EPOCH))"

  if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sacct -j "${SLURM_JOB_ID}" \
      --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode \
      > "${RUN_DIR}/slurm-sacct-in-job.txt" 2>&1
  fi

  if [[ -n "${REPORT_PYTHON}" ]]; then
    "${REPORT_PYTHON}" "${SCRIPT_DIR}/gpu_report.py" \
      --run-dir "${RUN_DIR}" \
      --exit-code "${status}" \
      --runner-elapsed-seconds "${elapsed_seconds}" \
      --python-module "${LOADED_PYTHON_MODULE}" || true
  else
    printf 'No Python interpreter was available to write REPORT.md.\n'
  fi

  printf 'Run artifacts: %s\n' "${RUN_DIR}"
  exit "${status}"
}
trap finalize EXIT

if ! type module >/dev/null 2>&1; then
  # Lmod initialization on Purdue RCAC hosts.
  source /etc/profile.d/modules.sh
fi

module --force purge
REQUESTED_PYTHON_MODULE="${GRPO_PYTHON_MODULE:-${SCHOLAR_CONDA_MODULE:-}}"
if [[ -n "${REQUESTED_PYTHON_MODULE}" ]]; then
  PYTHON_MODULE_CANDIDATES=("${REQUESTED_PYTHON_MODULE}")
else
  # The first entry is the current Scholar Python 3.12 module. The second is
  # retained as a compatibility fallback for older module trees.
  PYTHON_MODULE_CANDIDATES=(
    "anaconda/2024.10-py312"
    "conda/2024.09-py312"
  )
fi

for candidate in "${PYTHON_MODULE_CANDIDATES[@]}"; do
  if module load "${candidate}"; then
    LOADED_PYTHON_MODULE="${candidate}"
    break
  fi
done
if [[ "${LOADED_PYTHON_MODULE}" == "not loaded" ]]; then
  printf 'Could not load a supported Python 3.12 module. Available Anaconda modules:\n' >&2
  module spider anaconda >&2 || true
  printf 'Set GRPO_PYTHON_MODULE to an available Python 3.12 module and retry.\n' >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  printf 'Module %s did not provide conda.\n' "${LOADED_PYTHON_MODULE}" >&2
  exit 2
fi
eval "$(conda shell.bash hook)"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  conda create --prefix "${ENV_DIR}" python=3.12 pip -y
fi
conda activate "${ENV_DIR}"
REPORT_PYTHON="$(command -v python)"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

export HF_HOME="${CACHE_DIR}/huggingface"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TMPDIR="${WORK_ROOT}/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1'
python -m pip install --requirement "${SCRIPT_DIR}/requirements-gpu.txt"

python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip freeze --all > "${RUN_DIR}/environment.txt"
nvidia-smi -q > "${RUN_DIR}/nvidia-smi.txt"
sha256sum \
  "${SCRIPT_DIR}/gpu_smoke.py" \
  "${SCRIPT_DIR}/gpu_report.py" \
  "${SCRIPT_DIR}/requirements-gpu.txt" \
  "${SCRIPT_DIR}/run_scholar_gpu.sh" \
  "${SCRIPT_DIR}/scholar_gpu.sub" \
  > "${RUN_DIR}/source_manifest.sha256"

{
  printf 'utc_started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'hostname=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'kernel=%s\n' "$(uname -a)"
  printf 'python_module=%s\n' "${LOADED_PYTHON_MODULE}"
  printf 'python=%s\n' "$(python --version 2>&1)"
  printf 'conda=%s\n' "$(conda --version 2>&1)"
  env | LC_ALL=C sort | grep '^SLURM_' || true
} > "${RUN_DIR}/system-environment.txt"

python "${SCRIPT_DIR}/gpu_smoke.py" --output-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/run.log"
