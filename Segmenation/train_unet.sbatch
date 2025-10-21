#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 04:00:00
#SBATCH -J seg-unet
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out

set -euo pipefail

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SEG_DIR="$REPO_DIR/Segmenation"
PYFILE="$SEG_DIR/code/unet.py"
LOGDIR="$SEG_DIR/logs"
MODEL_DIR="${MODEL_DIR:-$SEG_DIR/models}"
VENV="${VENV_PATH:-$REPO_DIR/Super_resolution/.venv/bin/activate}"

mkdir -p "$LOGDIR" "$MODEL_DIR"

if [[ ! -f "$PYFILE" ]]; then
  echo "[error] Segmentation training script not found at $PYFILE" >&2
  exit 1
fi

if [[ ! -f "$VENV" ]]; then
  echo "[error] Virtualenv missing at $VENV" >&2
  exit 1
fi

source "$VENV"

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

EXTRA_ARGS="${EXTRA_ARGS:-}"

python - <<'PY'
import sys, tensorflow as tf
print("python:", sys.executable)
print("tf:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
PY

ts="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$LOGDIR/run-${ts}.log"

echo "[run] python -u $PYFILE ${EXTRA_ARGS}"

CMD=(python -u "$PYFILE")
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  CMD+=($EXTRA_ARGS)
fi

"${CMD[@]}" 2>&1 | tee "$LOGFILE"

echo "Finished at: $(date)"
