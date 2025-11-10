#!/usr/bin/env bash

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH --time=12:00:00
#SBATCH -J unet-seg-tune
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out

set -euo pipefail

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SEG_DIR="$REPO_DIR/Segmenation"
PY_SCRIPT="$SEG_DIR/code/unet_vinillia_optuna.py"
PYTHON_BIN="$REPO_DIR/Super_resolution/.venv/bin/python"
VENV_ACTIVATE="$REPO_DIR/Super_resolution/.venv/bin/activate"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] Optuna tuner not found at $PY_SCRIPT" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
if command -v module >/dev/null 2>&1; then
  module purge
  if [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE"
  else
    module load cuda >/dev/null 2>&1 || true
  fi
  if [[ -n "${CUDNN_MODULE:-}" ]]; then
    module load "$CUDNN_MODULE"
  fi
  module -t list 2>&1 || true
fi

if [[ -d "$CUDA_HOME/bin" ]]; then
  PATH="$CUDA_HOME/bin:${PATH:-}"
fi
if [[ -d "$CUDA_HOME/lib64" ]]; then
  LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH LD_LIBRARY_PATH

export PYTHONPATH="$SEG_DIR/code:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

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

if [[ ${#DATASET_PATHS[@]} -lt 5 ]]; then
  echo "[error] Unable to resolve dataset paths from dataset_paths.py" >&2
  exit 1
fi

TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-${DATASET_PATHS[0]}}"
TRAIN_MASK_DIR="${TRAIN_MASK_DIR:-${DATASET_PATHS[1]}}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-${DATASET_PATHS[2]}}"
VAL_MASK_DIR="${VAL_MASK_DIR:-${DATASET_PATHS[3]}}"
DEFAULT_MODEL_DIR="${DATASET_PATHS[4]}"

for dir_path in "$TRAIN_IMAGE_DIR" "$TRAIN_MASK_DIR" "$VAL_IMAGE_DIR" "$VAL_MASK_DIR"; do
  if [[ ! -d "$dir_path" ]]; then
    echo "[error] Dataset directory not found: $dir_path" >&2
    exit 1
  fi
done

TRIALS="${TRIALS:-20}"
TIMEOUT="${TIMEOUT:-}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LR_MIN="${LR_MIN:-1e-5}"
LR_MAX="${LR_MAX:-5e-4}"
DEPTH_MIN="${DEPTH_MIN:-3}"
DEPTH_MAX="${DEPTH_MAX:-5}"
BASE_CHANNELS_CHOICES="${BASE_CHANNELS_CHOICES:-32,48,64}"
BATCH_SIZE_CHOICES="${BATCH_SIZE_CHOICES:-4,8,12}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"
AUGMENT="${AUGMENT:-0}"
TUNE_AUGMENT="${TUNE_AUGMENT:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
FIT_VERBOSE="${FIT_VERBOSE:-2}"
STUDY_NAME="${STUDY_NAME:-unet_optuna}"
STORAGE="${STORAGE:-}"
LOAD_IF_EXISTS="${LOAD_IF_EXISTS:-1}"
RESULTS_PATH="${RESULTS_PATH:-$SEG_DIR/logs/optuna/${STUDY_NAME:-unet_optuna}.json}"
SAVE_BEST_MODEL="${SAVE_BEST_MODEL:-1}"
BEST_MODEL_DIR="${BEST_MODEL_DIR:-$DEFAULT_MODEL_DIR/optuna_best}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

LOG_ROOT="${LOG_ROOT:-$SEG_DIR/logs/optuna}"
mkdir -p "$LOG_ROOT" "$(dirname "$RESULTS_PATH")" "$BEST_MODEL_DIR"

ts="$(date +%Y%m%d-%H%M%S)"
run_tag="${STUDY_NAME:-optuna}_${ts}"
RUN_LOG="$LOG_ROOT/tune-${run_tag}.log"

CMD=(
  "$PYTHON_BIN" -u "$PY_SCRIPT"
  --train_image_dir "$TRAIN_IMAGE_DIR"
  --train_mask_dir "$TRAIN_MASK_DIR"
  --val_image_dir "$VAL_IMAGE_DIR"
  --val_mask_dir "$VAL_MASK_DIR"
  --image_size "$IMAGE_SIZE"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --trials "$TRIALS"
  --seed "$SEED"
  --lr_min "$LR_MIN"
  --lr_max "$LR_MAX"
  --depth_min "$DEPTH_MIN"
  --depth_max "$DEPTH_MAX"
  --base_channels_choices "$BASE_CHANNELS_CHOICES"
  --batch_size_choices "$BATCH_SIZE_CHOICES"
  --results_path "$RESULTS_PATH"
  --best_model_dir "$BEST_MODEL_DIR"
  --fit_verbose "$FIT_VERBOSE"
)

if [[ -n "$TIMEOUT" ]]; then
  CMD+=(--timeout "$TIMEOUT")
fi
if [[ -n "$LIMIT_TRAIN" ]]; then
  CMD+=(--limit_train "$LIMIT_TRAIN")
fi
if [[ -n "$LIMIT_VAL" ]]; then
  CMD+=(--limit_val "$LIMIT_VAL")
fi
if [[ "$AUGMENT" != "0" ]]; then
  CMD+=(--augment)
fi
if [[ "$TUNE_AUGMENT" != "0" ]]; then
  CMD+=(--tune_augment)
fi
if [[ "$MIXED_PRECISION" != "0" ]]; then
  CMD+=(--mixed_precision)
fi
if [[ -n "$STUDY_NAME" ]]; then
  CMD+=(--study_name "$STUDY_NAME")
fi
if [[ -n "$STORAGE" ]]; then
  CMD+=(--storage "$STORAGE")
fi
if [[ "$LOAD_IF_EXISTS" != "0" ]]; then
  CMD+=(--load_if_exists)
fi
if [[ "$SAVE_BEST_MODEL" != "0" ]]; then
  CMD+=(--save_best_model)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi

echo "[optuna-config] study=${STUDY_NAME:-""} trials=$TRIALS results=$RESULTS_PATH log=$RUN_LOG"
echo "[optuna-run] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo "Finished at: $(date)"
