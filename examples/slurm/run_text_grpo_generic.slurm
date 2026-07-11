#!/usr/bin/env bash
#SBATCH --job-name=evograph-text
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root, or override REPO_ROOT.
REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

ENV_PREFIX="${ENV_PREFIX:-/path/to/conda/envs/evograph-r1}"
CONDA_BASE="${CONDA_BASE:-$(dirname "$(dirname "$ENV_PREFIX")")}" 
DATASET="${DATASET:-2WikiMultiHopQA}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-Qwen2.5-3B-Instruct}"
TEXT_API_PORT="${TEXT_API_PORT:-8001}"

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
if ! conda activate "$ENV_PREFIX"; then
  echo "Warning: conda activate failed; falling back to the environment prefix." >&2
fi
export PATH="$ENV_PREFIX/bin:$PATH"
PYTHON_BIN="$ENV_PREFIX/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'

export SEARCH_API_URL="http://127.0.0.1:${TEXT_API_PORT}/search"
export TEXT_SEARCH_API_URL="$SEARCH_API_URL"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

"$PYTHON_BIN" script_api.py \
  --data_source "$DATASET" \
  --port "$TEXT_API_PORT" \
  --reload_interval 300 \
  --check_interval 60 \
  > "logs/text-api-${SLURM_JOB_ID:-manual}.log" 2>&1 &
TEXT_API_PID=$!

cleanup() {
  kill "$TEXT_API_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_api() {
  local timeout="${API_STARTUP_TIMEOUT:-300}"
  local interval="${API_HEALTHCHECK_INTERVAL:-2}"
  local deadline=$((SECONDS + timeout))
  local log_path="logs/text-api-${SLURM_JOB_ID:-manual}.log"

  while (( SECONDS < deadline )); do
    if ! kill -0 "$TEXT_API_PID" 2>/dev/null; then
      echo "Text API exited before becoming healthy." >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:${TEXT_API_PORT}/health" 2>/dev/null \
      | "$PYTHON_BIN" -c 'import json, sys; raise SystemExit(json.load(sys.stdin).get("status") != "healthy")'
    then
      echo "Text API is healthy."
      return 0
    fi
    sleep "$interval"
  done

  echo "Timed out after ${timeout}s waiting for text API." >&2
  tail -n 50 "$log_path" >&2 || true
  return 1
}

wait_for_api

bash run_grpo.sh \
  -p "$POLICY_MODEL_PATH" \
  -m "$POLICY_MODEL_NAME" \
  -d "$DATASET" \
  trainer.logger=['console'] \
  trainer.n_gpus_per_node="${SLURM_GPUS_ON_NODE:-4}"
