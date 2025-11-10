#!/usr/bin/env bash

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=31G
#SBATCH --time=12:00:00
#SBATCH -J unet-train-simple
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-%x-%j.out

set -euo pipefail

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SR_DIR="$REPO_DIR/Super_resolution"
PY_SCRIPT="$SR_DIR/code/train_adaptive_unet.py"
PYTHON_BIN="$SR_DIR/.venv/bin/python"
VENV_ACTIVATE="$SR_DIR/.venv/bin/activate"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python interpreter not found at $PYTHON_BIN" >&2
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

export PYTHONPATH="$SR_DIR/code:${PYTHONPATH:-}"

readarray -t DATASET_PATHS < <("$PYTHON_BIN" - <<'PY'
from dataset_paths import HR_TRAIN_DIR, LR_TRAIN_DIR, MODEL_ROOT, LOG_ROOT
print(HR_TRAIN_DIR)
print(LR_TRAIN_DIR)
print(MODEL_ROOT)
print(LOG_ROOT)
PY
)

if [[ ${#DATASET_PATHS[@]} -lt 4 ]]; then
  echo "[error] Unable to resolve dataset paths from dataset_paths.py" >&2
  exit 1
fi

HR_DIR="${HR_DIR:-${DATASET_PATHS[0]}}"
MODEL_DIR="${MODEL_DIR:-${DATASET_PATHS[2]}}"
LOG_DIR="${LOG_DIR:-${DATASET_PATHS[3]}}"

SCALE="${SCALE:-0.5}"
PATCH_SIZE="${PATCH_SIZE:-256}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-3}"
VAL_SPLIT="${VAL_SPLIT:-0.05}"
TEST_SPLIT="${TEST_SPLIT:-0.05}"
SEED="${SEED:-1234}"
MIXED_PRECISION="${MIXED_PRECISION:-1}"
RUN_NAME="${RUN_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$SR_DIR/logs/experiment1/evaluation}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
SUMMARY_ARCHIVE_DIR="${SUMMARY_ARCHIVE_DIR:-$SR_DIR/logs/experiment_1}"

mkdir -p "$MODEL_DIR" "$LOG_DIR"

if [[ ! -d "$HR_DIR" ]]; then
  echo "[error] High-resolution directory not found: $HR_DIR" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[error] Training entry point not found at $PY_SCRIPT" >&2
  exit 1
fi

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

ts="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$LOG_DIR/run-simple-scale${SCALE}-${ts}.log"

echo "[simple-config] scale=$SCALE hr_dir=$HR_DIR model_dir=$MODEL_DIR log_dir=$LOG_DIR"

CMD=("$PYTHON_BIN" -u "$PY_SCRIPT"
  --scale "$SCALE"
  --high_res_dir "$HR_DIR"
  --patch_size "$PATCH_SIZE"
  --patches_per_image "$PATCHES_PER_IMAGE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --val_split "$VAL_SPLIT"
  --test_split "$TEST_SPLIT"
  --seed "$SEED"
  --model_dir "$MODEL_DIR"
  --log_dir "$LOG_DIR"
)

if [[ "$MIXED_PRECISION" != "0" ]]; then
  CMD+=(--mixed_precision)
fi

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi

if [[ -n "${INITIAL_EPOCH:-}" ]]; then
  CMD+=(--initial_epoch "$INITIAL_EPOCH")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi

DEPTH_OVERRIDE_VALUE=""
if [[ "${EXTRA_ARGS:-}" =~ --depth_override[[:space:]]*([0-9]+) ]]; then
  DEPTH_OVERRIDE_VALUE="${BASH_REMATCH[1]}"
elif [[ "${EXTRA_ARGS:-}" =~ --depth_override=([0-9]+) ]]; then
  DEPTH_OVERRIDE_VALUE="${BASH_REMATCH[1]}"
fi
DEPTH_OVERRIDE_ARG=()
if [[ -n "$DEPTH_OVERRIDE_VALUE" ]]; then
  DEPTH_OVERRIDE_ARG=(--depth-override "$DEPTH_OVERRIDE_VALUE")
fi

echo "[simple-run] ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$RUN_LOG"

if [[ -n "$SUMMARY_ARCHIVE_DIR" ]]; then
  mkdir -p "$SUMMARY_ARCHIVE_DIR"
  summary_path="$(ls -t "$LOG_DIR"/*/model_summary.txt 2>/dev/null | head -n 1 || true)"
  if [[ -n "$summary_path" ]]; then
    summary_basename="$(basename "$(dirname "$summary_path")")_model_summary.txt"
    cp "$summary_path" "$SUMMARY_ARCHIVE_DIR/$summary_basename"
  else
    echo "[warn] No model_summary.txt found under $LOG_DIR to archive." >&2
  fi
fi

mkdir -p "$EVAL_OUTPUT_DIR"
checkpoint_path="$(ls -t "$MODEL_DIR"/unet_adaptive_scale_new_loss*.keras 2>/dev/null | head -n 1 || true)"
if [[ -z "$checkpoint_path" ]]; then
  echo "[warn] No checkpoint found in $MODEL_DIR after training; skipping evaluation." >&2
else
  checkpoint_base="$(basename "$checkpoint_path" .keras)"
  eval_run_name="${RUN_NAME:-$checkpoint_base}_eval"
  eval_log="${RUN_LOG}.eval"
  EVAL_CMD=(
    "$PYTHON_BIN" -u "$SR_DIR/code/evaluate_model.py"
    --model-path "$checkpoint_path"
    --scale "$SCALE"
    --patch-size "$PATCH_SIZE"
    --batch-size "$EVAL_BATCH_SIZE"
    --output-dir "$EVAL_OUTPUT_DIR"
    --run-name "$eval_run_name"
  )
  if [[ ${#DEPTH_OVERRIDE_ARG[@]} -gt 0 ]]; then
    EVAL_CMD+=("${DEPTH_OVERRIDE_ARG[@]}")
  fi
  echo "[simple-eval] ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}" 2>&1 | tee "$eval_log"
fi

echo "Finished at: $(date)"
