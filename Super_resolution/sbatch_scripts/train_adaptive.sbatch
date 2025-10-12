#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 04:00:00
#SBATCH -J unet-train
#SBATCH -o sbatch_scripts/../logs/slurm-%x-%j.out

set -euo pipefail

echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU(s):     $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

# ─── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYFILE="$REPO_DIR/code/train_adaptive_unet.py"
LOGDIR="$REPO_DIR/logs"
SCRATCH_ROOT="/scratch/$USER/Super_resolution"

mkdir -p "$LOGDIR" "$SCRATCH_ROOT"

# ─── GPU modules (optional) ───────────────────────────────────────────────────
module purge

if [[ -n "${CUDA_MODULE:-}" ]]; then
  module load "$CUDA_MODULE"
else
  echo "[info] CUDA_MODULE not set; skipping explicit CUDA module load."
fi

if [[ -n "${CUDNN_MODULE:-}" ]]; then
  module load "$CUDNN_MODULE"
else
  echo "[info] CUDNN_MODULE not set; skipping explicit cuDNN module load."
fi

module -t list 2>&1 || true

# ─── Export library paths if provided by modules ──────────────────────────────
if [[ -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="$EBROOTCUDA"
  export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${EBROOTCUDNN:-}" ]]; then
  export LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ─── Python / Env ──────────────────────────────────────────────────────────────
source "$REPO_DIR/.venv/bin/activate"

# ─── Runtime env tweaks (safe defaults for TF) ─────────────────────────────────
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

echo "[data] scratch: /scratch/$USER"
du -sh "/scratch/$USER" 2>/dev/null || true

# ─── Run configuration (override via env vars) ────────────────────────────────
SCALE="${SCALE:-0.6}"
HR_DIR="${HR_DIR:-$REPO_DIR/dataset/Raw Data/high_res}"
LR_DIR="${LR_DIR:-$REPO_DIR/dataset/Raw Data/low_res}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
USE_MIXED_PRECISION="${MIXED_PRECISION:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$USE_MIXED_PRECISION" == "0" ]]; then
  MIXED_FLAG=""
else
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
