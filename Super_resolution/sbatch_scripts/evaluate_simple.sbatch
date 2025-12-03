#!/usr/bin/env bash

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH --time=01:00:00
#SBATCH -J unet-eval-simple
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out


# ------------------------------- Environment setup and checks -------------------------------

set -euo pipefail

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SR_DIR="$REPO_DIR/Super_resolution"
PY_SCRIPT="$SR_DIR/code/evaluate_model.py"
PYTHON_BIN="$SR_DIR/.venv/bin/python"
VENV_ACTIVATE="$SR_DIR/.venv/bin/activate"

# Check if Python interpreter exists
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

# Activate virtual environment
source "$VENV_ACTIVATE"

# Setup CUDA environment assuming its in /opt/cuda
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
# Cluster specific CUDA module loading
if command -v module >/dev/null 2>&1; then
  # Unload any existing modules to ensure clean environment so previous job doesn't affect this one
  module purge
  # Load CUDA module if specified, otherwise load default CUDA module
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  else
    module load cuda >/dev/null 2>&1 || true
  fi
  # Load cuDNN module if specified, otherwise load default cuDNN module
  if [[ -n "${CUDNN_MODULE:-}" ]]; then
    module load "$CUDNN_MODULE"
  fi
  # List loaded modules for debugging
  module -t list 2>&1 || true
fi

# Add CUDA bin and lib64 to PATH and LD_LIBRARY_PATH
if [[ -d "$CUDA_HOME/bin" ]]; then
  PATH="$CUDA_HOME/bin:${PATH:-}"
fi
if [[ -d "$CUDA_HOME/lib64" ]]; then
  LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ------------------------------- Configuration -------------------------------

export PATH LD_LIBRARY_PATH
export PYTHONPATH="$SR_DIR/code:${PYTHONPATH:-}"

# Defaults
SCALE="${SCALE:-0.5}"
PATCH_SIZE="${PATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SR_DIR/logs/evaluation}"
RUN_NAME="${RUN_NAME:-eval_run}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "[error] MODEL_PATH must be provided." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ----------------------------- Command to run evaluation script -----------------------------
CMD=("$PYTHON_BIN" -u "$PY_SCRIPT"
  --model-path "$MODEL_PATH"
  --scale "$SCALE"
  --patch-size "$PATCH_SIZE"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
)

# ----------------------------- Depth Override -----------------------------

DEPTH_OVERRIDE_VALUE=""
if [[ "${EXTRA_ARGS:-}" =~ --depth_override[[:space:]]*([0-9]+) ]]; then
  DEPTH_OVERRIDE_VALUE="${BASH_REMATCH[1]}"
elif [[ "${EXTRA_ARGS:-}" =~ --depth_override=([0-9]+) ]]; then
  DEPTH_OVERRIDE_VALUE="${BASH_REMATCH[1]}"
fi
if [[ -n "$DEPTH_OVERRIDE_VALUE" ]]; then
  CMD+=(--depth-override "$DEPTH_OVERRIDE_VALUE")
fi

echo "[eval-run] ${CMD[*]}"
"${CMD[@]}"
