#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${GRPO_SMOKE_WORK_ROOT:-${HOME}/code/sysllmatic-grpo-smoke}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${WORK_ROOT}/.venv"
BOOTSTRAP_DIR="${WORK_ROOT}/bootstrap"
CACHE_DIR="${WORK_ROOT}/cache"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${WORK_ROOT}/runs/${RUN_ID}"

mkdir -p "${WORK_ROOT}" "${CACHE_DIR}/huggingface" "${CACHE_DIR}/pip" "${RUN_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]] || \
   ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  if ! python3 -m venv "${VENV_DIR}"; then
    # Minimal Ubuntu/WSL images can omit ensurepip. Bootstrap virtualenv into
    # the experiment directory so the smoke test remains isolated and does
    # not require sudo or a system-package change.
    python3 -m pip install \
      --target "${BOOTSTRAP_DIR}" \
      'virtualenv==20.31.2'
    PYTHONPATH="${BOOTSTRAP_DIR}" python3 -m virtualenv --clear "${VENV_DIR}"
  fi
fi

export HF_HOME="${CACHE_DIR}/huggingface"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.7.1'
"${VENV_DIR}/bin/python" -m pip install \
  --requirement "${SCRIPT_DIR}/requirements-cpu.txt"

"${VENV_DIR}/bin/python" -m pip freeze --all > "${RUN_DIR}/environment.txt"

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/cpu_preflight.py" \
  --output-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/run.log"

printf '%s\n' "${RUN_DIR}"
