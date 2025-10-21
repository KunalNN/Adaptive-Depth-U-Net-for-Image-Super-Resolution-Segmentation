#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 04:00:00
#SBATCH -J unet-train
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out

set -euo pipefail

echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU(s):     $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SR_DIR="$REPO_DIR/Super_resolution"
PYFILE="$SR_DIR/code/train_adaptive_unet.py"
LOGDIR="$SR_DIR/logs"
VENV="$SR_DIR/.venv/bin/activate"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -d "$CUDA_HOME/lib64" && ":$LD_LIBRARY_PATH:" != *":$CUDA_HOME/lib64:"* ]]; then
  LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
PATH="$CUDA_HOME/bin:${PATH:-}"
export CUDA_HOME PATH LD_LIBRARY_PATH

SCRATCH_ROOT="/scratch/$USER/Super_resolution"
SCRATCH_DATA_ROOT="$SCRATCH_ROOT/Final_data/Super_resolution"
CANONICAL_DATA_ROOT="${CANONICAL_DATA_ROOT:-/vol/csedu-nobackup/project/cseduproject/Final_data/Super_resolution}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH_DATA_ROOT}"
SYNC_FROM_SHARED="${SYNC_FROM_SHARED:-1}"

mkdir -p "$LOGDIR" "$SCRATCH_ROOT" "$SCRATCH_ROOT/models" "$DATA_ROOT"

if [[ ! -f "$PYFILE" ]]; then
  echo "[error] Training script not found at $PYFILE" >&2
  exit 1
fi

# Load GPU stack via environment modules when available so cuBLAS/cuDNN are discoverable.
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

# Stage shared dataset into node-local scratch for faster training I/O on each node.
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

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] Data root not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$VENV" ]]; then
  echo "[error] Virtualenv missing: $VENV" >&2
  exit 1
fi

source "$VENV"

# If CUDA/cuDNN wheels are installed via pip (tensorflow[and-cuda]), expose them.
PIP_CUDA_PATHS="$(python - <<'PY'
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

# ─── Runtime env tweaks (safe defaults for TF) ─────────────────────────────────
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

# ─── Run configuration (override via env vars) ────────────────────────────────
SCALE="${SCALE:-0.6}"
HR_DIR="${HR_DIR:-$DATA_ROOT/DIV2K_train_HR}"
LR_DIR="${LR_DIR:-$DATA_ROOT/DIV2K_train_LR_bicubic-2/X4}"
MODEL_DIR="${MODEL_DIR:-$SCRATCH_ROOT/models}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
USE_MIXED_PRECISION="${MIXED_PRECISION:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "$HR_DIR" ]]; then
  echo "[error] High-resolution directory not found: $HR_DIR" >&2
  exit 1
fi

if [[ ! -d "$LR_DIR" ]]; then
  echo "[error] Low-resolution directory not found: $LR_DIR" >&2
  exit 1
fi

MIXED_FLAG=""
if [[ "$USE_MIXED_PRECISION" != "0" ]]; then
  MIXED_FLAG="--mixed_precision"
fi

mkdir -p "$MODEL_DIR"

# ─── Sanity info ───────────────────────────────────────────────────────────────
python - <<'PY'
import sys, tensorflow as tf
print("python:", sys.executable)
print("tf:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
PY

# ─── Train ─────────────────────────────────────────────────────────────────────
ts="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$LOGDIR/run-scale${SCALE}-${ts}.log"

echo "[config] scale=$SCALE hr_dir=$HR_DIR lr_dir=$LR_DIR model_dir=$MODEL_DIR"
echo "[run] python -u $PYFILE --scale $SCALE ..."

CMD=(python -u "$PYFILE"
  --scale "$SCALE"
  --high_res_dir "$HR_DIR"
  --low_res_dir "$LR_DIR"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --model_dir "$MODEL_DIR"
)

if [[ -n "$MIXED_FLAG" ]]; then
  CMD+=("$MIXED_FLAG")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi

"${CMD[@]}" 2>&1 | tee "$LOGFILE"

echo "Finished at: $(date)"
