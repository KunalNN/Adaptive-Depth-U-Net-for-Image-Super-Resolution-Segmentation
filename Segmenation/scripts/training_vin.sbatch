#!/usr/bin/env bash

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH --time=12:00:00
#SBATCH -J unet-seg-train
#SBATCH --chdir=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation
#SBATCH --output=Segmenation/logs/log_vinillia/slurm-%x-%j.out
#SBATCH --error=Segmenation/logs/log_vinillia/slurm-%x-%j.out

# ------------------------------- Environment setup and checks -------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the directory where sbatch was invoked; fall back to REPO_DIR env or the
# script parent. This keeps the setup predictable without nested heuristics.
if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  else
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  fi
fi

if [[ ! -d "$REPO_DIR/Segmenation" || ! -d "$REPO_DIR/Super_resolution" ]]; then
  PARENT_REPO="$(cd "$REPO_DIR/.." && pwd)"
  if [[ -d "$PARENT_REPO/Segmenation" && -d "$PARENT_REPO/Super_resolution" ]]; then
    REPO_DIR="$PARENT_REPO"
  else
    echo "[error] Unable to locate repo root (checked $REPO_DIR and $PARENT_REPO). Set REPO_DIR when submitting the job." >&2
    exit 1
  fi
fi
SEG_DIR="$REPO_DIR/Segmenation"
PY_SCRIPT="${PY_SCRIPT:-$SEG_DIR/code/unet_vinillia.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SEG_DIR/code/evaluate_adaptive_unet.py}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/Super_resolution/.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO_DIR/Super_resolution/.venv/bin/activate}"

# Check if Python interpreter exists
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] Training entry point not found at $PY_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[error] Evaluation entry point not found at $EVAL_SCRIPT" >&2
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
export PATH LD_LIBRARY_PATH

# ------------------------------- Dataset and Configuration -------------------------------

export PYTHONPATH="$SEG_DIR/code:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# Get dataset paths from dataset_paths.py 
readarray -t DATASET_PATHS < <("$PYTHON_BIN" - <<'PY'
from dataset_paths import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    MODEL_ROOT,
)
for path in (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    MODEL_ROOT,
):
    print(path)
PY
)

# Verify dataset paths exist
if [[ ${#DATASET_PATHS[@]} -lt 5 ]]; then
  echo "[error] Unable to resolve dataset paths from dataset_paths.py" >&2
  exit 1
fi

# Set default values for dataset paths
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-${DATASET_PATHS[0]}}"
TRAIN_MASK_DIR="${TRAIN_MASK_DIR:-${DATASET_PATHS[1]}}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-${DATASET_PATHS[2]}}"
VAL_MASK_DIR="${VAL_MASK_DIR:-${DATASET_PATHS[3]}}"
MODEL_DIR="${MODEL_DIR:-${DATASET_PATHS[4]}}"
LOG_DIR="${LOG_DIR:-$SEG_DIR/logs}"

# Create model and log directories
mkdir -p "$MODEL_DIR" "$LOG_DIR"

# Verify directories exist
for dir_path in "$TRAIN_IMAGE_DIR" "$TRAIN_MASK_DIR" "$VAL_IMAGE_DIR" "$VAL_MASK_DIR"; do
  if [[ ! -d "$dir_path" ]]; then
    echo "[error] Dataset directory not found: $dir_path" >&2
    exit 1
  fi
done

# Set default values for Hyperparameters
BATCH_SIZE="${BATCH_SIZE:-0}"
EPOCHS="${EPOCHS:-0}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
DEPTH="${DEPTH:-4}"
PROTOCOL="${PROTOCOL:-A}"
RUN_NAME="${RUN_NAME:-}"

# ------------------------------- CUDA Environment Setup -------------------------------

# Get CUDA paths from Python environment robust fallback if system CUDA is missing 
PIP_CUDA_PATHS="$("$PYTHON_BIN" - <<'PY'
import os, site
paths = []
for root in site.getsitepackages():
    for subdir in (
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cudnn/lib",
        "nvidia/cufft/lib",
        "nvidia/curand/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "nvidia/nccl/lib",
        "nvidia/nvjitlink/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_nvcc/lib",
        "nvidia/cuda_cupti/lib",
    ):
        candidate = os.path.join(root, subdir)
        if os.path.isdir(candidate):
            paths.append(candidate)
if paths:
    seen = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    print(":".join(seen))
PY
)"
if [[ -n "$PIP_CUDA_PATHS" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$PIP_CUDA_PATHS:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$PIP_CUDA_PATHS"
  fi
fi

# Timestamp for run log
ts="$(date +%Y%m%d-%H%M%S)"
run_tag="${RUN_NAME:-unet}"
run_tag="${run_tag// /_}"
RUN_LOG="$LOG_DIR/train-${run_tag}-${ts}.log"
EVAL_LOG="$LOG_DIR/eval-${run_tag}-${ts}.log"
CKPT_PATH="$MODEL_DIR/$RUN_NAME.weights.h5"
TRAIN_CONFIG="$LOG_DIR/$RUN_NAME/config.json"

# Print configuration
echo "[seg-config] protocol=$PROTOCOL images=$TRAIN_IMAGE_DIR masks=$TRAIN_MASK_DIR model_dir=$MODEL_DIR log_dir=$LOG_DIR"

# ----------------------------- Command to run training script -----------------------------
CMD=(
  "$PYTHON_BIN" -u "$PY_SCRIPT"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --base_channels "$BASE_CHANNELS"
  --depth "$DEPTH"
  --model_dir "$MODEL_DIR"
  --train_image_dir "$TRAIN_IMAGE_DIR"
  --train_mask_dir "$TRAIN_MASK_DIR"
  --val_image_dir "$VAL_IMAGE_DIR"
  --val_mask_dir "$VAL_MASK_DIR"
)

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

echo "[seg-run] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo "Finished at: $(date)"
