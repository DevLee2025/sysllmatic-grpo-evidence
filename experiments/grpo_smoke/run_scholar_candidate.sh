#!/usr/bin/env bash
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module --force purge
REQUESTED_PYTHON_MODULE="${GRPO_PYTHON_MODULE:-${SCHOLAR_CONDA_MODULE:-}}"
if [[ -n "${REQUESTED_PYTHON_MODULE}" ]]; then
  PYTHON_MODULE_CANDIDATES=("${REQUESTED_PYTHON_MODULE}")
else
  PYTHON_MODULE_CANDIDATES=(
    "anaconda/2024.10-py312"
    "conda/2024.09-py312"
  )
fi

LOADED_PYTHON_MODULE=""
for candidate in "${PYTHON_MODULE_CANDIDATES[@]}"; do
  if module load "${candidate}"; then
    LOADED_PYTHON_MODULE="${candidate}"
    break
  fi
done
if [[ -z "${LOADED_PYTHON_MODULE}" ]]; then
  printf 'Could not load a supported Python 3.12 module. Set GRPO_PYTHON_MODULE and retry.\n' >&2
  module spider anaconda >&2 || true
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  printf 'Module %s did not provide conda.\n' "${LOADED_PYTHON_MODULE}" >&2
  exit 2
fi
eval "$(conda shell.bash hook)"

: "${RCAC_SCRATCH:?RCAC_SCRATCH is not set; run this script on Scholar}"
WORK_ROOT="${GRPO_SMOKE_WORK_ROOT:-${RCAC_SCRATCH}/sysllmatic-grpo-smoke}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${WORK_ROOT}/envs/trl-py312"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  printf 'Missing %s. Run run_scholar_gpu.sh once to create the pinned environment.\n' "${ENV_DIR}" >&2
  exit 2
fi

conda activate "${ENV_DIR}"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
export HF_HOME="${WORK_ROOT}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

exec python "${SCRIPT_DIR}/generate_candidate.py" --require-cuda "$@"
