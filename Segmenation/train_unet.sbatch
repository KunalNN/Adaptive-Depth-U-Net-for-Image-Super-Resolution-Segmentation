#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 12:00:00
#SBATCH -J adaptive-unet
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out

set -euo pipefail

echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU(s):     $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SEG_DIR="$REPO_DIR/Segmenation"
PYFILE="$SEG_DIR/code/train_adaptive_unet.py"
LOGDIR="$SEG_DIR/logs"
VENV_ACTIVATE="$SEG_DIR/.venv/bin/activate"
PYTHON_BIN="$SEG_DIR/.venv/bin/python"

CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -d "$CUDA_HOME/lib64" && ":$LD_LIBRARY_PATH:" != *":$CUDA_HOME/lib64:"* ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$CUDA_HOME/bin:${PATH:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

SCRATCH_BASE="/scratch/$USER"
SCRATCH_DATA_ROOT="$SCRATCH_BASE/Final_data/Segmenation/ISIC-2017"
CANONICAL_DATA_ROOT="${CANONICAL_DATA_ROOT:-/vol/csedu-nobackup/project/cseduproject/Final_data/Segmenation/ISIC-2017}"

if [[ -n "${DATA_ROOT:-}" ]]; then
  USER_DATA_ROOT="$DATA_ROOT"
else
  DATA_ROOT="$SCRATCH_DATA_ROOT"
fi
SYNC_FROM_SHARED="${SYNC_FROM_SHARED:-1}"

readarray -t DATASET_PATHS < <(PYTHONPATH="$SEG_DIR/code:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
from dataset_paths import (
    DATA_ROOT,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    MODEL_ROOT,
    LOG_ROOT,
)
from pathlib import Path

print(Path(DATA_ROOT).resolve())
print(Path(TRAIN_IMAGE_DIR).relative_to(DATA_ROOT))
print(Path(TRAIN_MASK_DIR).relative_to(DATA_ROOT))
print(Path(VALID_IMAGE_DIR).relative_to(DATA_ROOT))
print(Path(VALID_MASK_DIR).relative_to(DATA_ROOT))
print(Path(MODEL_ROOT).resolve())
print(Path(LOG_ROOT).resolve())
PY
)

if [[ ${#DATASET_PATHS[@]} -lt 7 ]]; then
  echo "[error] Failed to resolve dataset paths via dataset_paths.py" >&2
  exit 1
fi

DEFAULT_DATA_ROOT="${DATASET_PATHS[0]}"
TRAIN_IMAGE_SUFFIX="${DATASET_PATHS[1]}"
TRAIN_MASK_SUFFIX="${DATASET_PATHS[2]}"
VAL_IMAGE_SUFFIX="${DATASET_PATHS[3]}"
VAL_MASK_SUFFIX="${DATASET_PATHS[4]}"
DEFAULT_MODEL_ROOT="${DATASET_PATHS[5]}"
DEFAULT_TB_ROOT="${DATASET_PATHS[6]}"

if [[ -n "${USER_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$USER_DATA_ROOT"
else
  DATA_ROOT="$DEFAULT_DATA_ROOT"
fi

TRAIN_IMAGE_DIR="$DATA_ROOT/$TRAIN_IMAGE_SUFFIX"
TRAIN_MASK_DIR="$DATA_ROOT/$TRAIN_MASK_SUFFIX"
VAL_IMAGE_DIR="$DATA_ROOT/$VAL_IMAGE_SUFFIX"
VAL_MASK_DIR="$DATA_ROOT/$VAL_MASK_SUFFIX"

mkdir -p "$LOGDIR" "$SCRATCH_BASE" "$DATA_ROOT" "$DEFAULT_MODEL_ROOT" "$DEFAULT_TB_ROOT"

if [[ ! -f "$PYFILE" ]]; then
  echo "[error] Training script not found at $PYFILE" >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  else
    echo "[warn] CUDA_MODULE not set; relying on system CUDA at $CUDA_HOME"
  fi
  if [[ -n "${CUDNN_MODULE:-}" ]]; then
    module load "$CUDNN_MODULE"
  else
    echo "[warn] CUDNN_MODULE not set; assuming cuDNN bundled with CUDA module."
  fi
  module -t list 2>&1 || true
fi

if [[ "$SYNC_FROM_SHARED" != "0" ]]; then
  if [[ ! -d "$CANONICAL_DATA_ROOT" ]]; then
    echo "[error] Shared data root not found: $CANONICAL_DATA_ROOT" >&2
    exit 1
  fi
  echo "[stage] Syncing data from $CANONICAL_DATA_ROOT to $DATA_ROOT"
  if ! srun --ntasks=1 rsync -a --delete "$CANONICAL_DATA_ROOT/" "$DATA_ROOT/"; then
    echo "[error] Data staging failed" >&2
    exit 1
  fi
fi

for path in "$TRAIN_IMAGE_DIR" "$TRAIN_MASK_DIR" "$VAL_IMAGE_DIR" "$VAL_MASK_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "[error] Required dataset directory missing: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[error] Virtualenv missing: $VENV_ACTIVATE" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

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
  export LD_LIBRARY_PATH="$PIP_CUDA_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

PROTOCOL="${PROTOCOL:-A}"
EPOCHS="${EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-0}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
DEPTH="${DEPTH:-4}"
PATIENCE="${PATIENCE_OVERRIDE:-}"
RUN_NAME="${RUN_NAME:-}"
MIXED_PRECISION="${MIXED_PRECISION:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  "--protocol" "$PROTOCOL"
  "--base_channels" "$BASE_CHANNELS"
  "--depth" "$DEPTH"
  "--train_images" "$TRAIN_IMAGE_DIR"
  "--train_masks" "$TRAIN_MASK_DIR"
  "--val_images" "$VAL_IMAGE_DIR"
  "--val_masks" "$VAL_MASK_DIR"
  "--model_dir" "$DEFAULT_MODEL_ROOT"
  "--log_dir" "$DEFAULT_TB_ROOT"
)

if [[ "$EPOCHS" != "0" ]]; then
  ARGS+=("--epochs" "$EPOCHS")
fi
if [[ "$BATCH_SIZE" != "0" ]]; then
  ARGS+=("--batch_size" "$BATCH_SIZE")
fi
if [[ -n "$PATIENCE" ]]; then
  ARGS+=("--patience" "$PATIENCE")
fi
if [[ -n "$RUN_NAME" ]]; then
  ARGS+=("--run_name" "$RUN_NAME")
fi
if [[ "$MIXED_PRECISION" == "1" ]]; then
  ARGS+=("--mixed_precision")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  ARGS+=("${EXTRA_ARR[@]}")
fi

echo "[info] Running training with arguments: ${ARGS[*]}"
"$PYTHON_BIN" "$PYFILE" "${ARGS[@]}"

echo "Finished at: $(date)"
