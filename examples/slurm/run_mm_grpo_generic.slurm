#!/usr/bin/env bash
#SBATCH --job-name=evograph-mm-grpo
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

mkdir -p logs

ENV_PREFIX="${ENV_PREFIX:-/path/to/conda/envs/evograph-r1}"
CONDA_BASE="${CONDA_BASE:-$(dirname "$(dirname "$ENV_PREFIX")")}" 
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-BAAI/bge-large-en-v1.5}"
MM_DATA_ROOT="${MM_DATA_ROOT:-$REPO_ROOT}"
MM_EMBEDDING_BACKEND="${MM_EMBEDDING_BACKEND:-gme}"
MM_EMBED_MODEL_PATH="${MM_EMBED_MODEL_PATH:-/path/to/gme-Qwen2-VL-2B-Instruct}"
MM_EMBED_DEVICE="${MM_EMBED_DEVICE:-cpu}"
TEXT_DATA_SOURCE="${TEXT_DATA_SOURCE:-${DATA_SOURCE:-}}"
MM_DATASET="${MM_DATASET:-}"
MM_SUBSET="${MM_SUBSET:-}"
RUN_ID="${SLURM_JOB_ID:-manual}"

if [ -z "$TEXT_DATA_SOURCE" ] || [ -z "$MM_DATASET" ] || [ -z "$MM_SUBSET" ]; then
  echo "Set TEXT_DATA_SOURCE, MM_DATASET, and MM_SUBSET before submitting." >&2
  exit 1
fi

if [ ! -d "$MM_EMBED_MODEL_PATH" ]; then
  echo "MM_EMBED_MODEL_PATH must point to a local GME or Qwen3-VL model directory." >&2
  exit 1
fi

TEXT_KB_DIR="${TEXT_KB_DIR:-expr/${TEXT_DATA_SOURCE}}"
MM_SOURCE_KB_DIR="${MM_SOURCE_KB_DIR:-${MM_DATA_ROOT}/expr_mm/${MM_DATASET}}"
MM_WORKING_DIR="${MM_WORKING_DIR:-${MM_DATA_ROOT}/expr_mm/${MM_DATASET}_working_${RUN_ID}}"

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

export POLICY_MODEL_PATH
export BGE_MODEL_PATH
export MM_DATA_ROOT
export MM_EMBEDDING_BACKEND
export MM_EMBED_MODEL_PATH
export MM_EMBED_DEVICE
export MM_SOURCE_KB_DIR
export EVOGRAPH_MM_WORKING_DIR="$MM_WORKING_DIR"
export TEXT_SEARCH_API_URL="${TEXT_SEARCH_API_URL:-http://127.0.0.1:8001/search}"
export MM_SEARCH_API_URL="${MM_SEARCH_API_URL:-http://127.0.0.1:8003/search}"
export WEBSEARCH_CACHE_ENABLED="${WEBSEARCH_CACHE_ENABLED:-true}"
export WEBSEARCH_CACHE_DIR="${WEBSEARCH_CACHE_DIR:-$REPO_ROOT/websearch_cache}"
export WEBSEARCH_CACHE_DATASET="${WEBSEARCH_CACHE_DATASET:-${MM_DATASET}_${MM_SUBSET}}"
export TOOL_USE_DEFERRED_EXECUTION="${TOOL_USE_DEFERRED_EXECUTION:-true}"
export TOOL_BATCH_SIZE="${TOOL_BATCH_SIZE:-50}"

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from evograph_mm.kb.graph_edit import prepare_edit_working_dir

prepare_edit_working_dir(
    Path(os.environ["MM_SOURCE_KB_DIR"]),
    Path(os.environ["EVOGRAPH_MM_WORKING_DIR"]),
)
PY

"$PYTHON_BIN" script_api.py \
  --data_source "$TEXT_DATA_SOURCE" \
  --working_dir "$TEXT_KB_DIR" \
  --port 8001 \
  > "logs/text-api-${RUN_ID}.log" 2>&1 &
TEXT_API_PID=$!

"$PYTHON_BIN" script_api_mm.py \
  --dataset "$MM_DATASET" \
  --subset "$MM_SUBSET" \
  --working_dir "$MM_WORKING_DIR" \
  --model "$MM_EMBED_MODEL_PATH" \
  --embedding-backend "$MM_EMBEDDING_BACKEND" \
  --port 8003 \
  --reload_interval 300 \
  --check_interval 60 \
  > "logs/mm-api-${RUN_ID}.log" 2>&1 &
MM_API_PID=$!

cleanup() {
  kill "$TEXT_API_PID" "$MM_API_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_api() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local log_path="$4"
  local timeout="${API_STARTUP_TIMEOUT:-900}"
  local interval="${API_HEALTHCHECK_INTERVAL:-5}"
  local deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming healthy." >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 5 "$url" 2>/dev/null \
      | "$PYTHON_BIN" -c 'import json, sys; raise SystemExit(json.load(sys.stdin).get("status") != "healthy")'
    then
      echo "$name is healthy."
      return 0
    fi
    sleep "$interval"
  done

  echo "Timed out after ${timeout}s waiting for $name." >&2
  tail -n 50 "$log_path" >&2 || true
  return 1
}

wait_for_api \
  "text API" \
  "http://127.0.0.1:8001/health" \
  "$TEXT_API_PID" \
  "logs/text-api-${RUN_ID}.log"
wait_for_api \
  "multimodal API" \
  "http://127.0.0.1:8003/health" \
  "$MM_API_PID" \
  "logs/mm-api-${RUN_ID}.log"

bash run_mm_grpo.sh \
  -p "$POLICY_MODEL_PATH" \
  -m "$(basename "$POLICY_MODEL_PATH")" \
  -d "$MM_DATASET" \
  -s "$MM_SUBSET"
