#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 03:00:00
#SBATCH -J unet-train
#SBATCH -o /home/knarwani/thesis/super-resolution/logs/slurm-%x-%j.out
## If you want email notifications, uncomment & edit the next line:
## #SBATCH --mail-type=BEGIN,END,FAIL
## #SBATCH --mail-user=you@cs.ru.nl

set -euo pipefail

echo "Job:        $SLURM_JOB_NAME"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU(s):     $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/home/knarwani/thesis/super-resolution"
PYFILE="$REPO_DIR/code/u-net-custom-scale.py"
LOGDIR="$REPO_DIR/logs"
MODELDIR="$REPO_DIR/models"


mkdir -p "$LOGDIR" "$MODELDIR"

source "/home/knarwani/thesis/super-resolution/.venv/bin/activate" 
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

# New: push caches and our base dir to scratch
export SR_BASE="/scratch/knarwani/super-resolution"   # <-- used by make_callbacks()
export TMPDIR="/scratch/knarwani/super-resolution/tmp"
export KERAS_HOME="/scratch/knarwani/.keras"


mkdir -p "$SR_BASE/models" "$SR_BASE/runs" "$TMPDIR" "$KERAS_HOME"

# ─── Sanity info ───────────────────────────────────────────────────────────────
python - <<'PY'
import sys, tensorflow as tf
print("python:", sys.executable)
print("tf:", tf.__version__)
print("gpus:", tf.config.list_physical_devices("GPU"))
PY

# ─── Train ─────────────────────────────────────────────────────────────────────
ts="$(date +%Y%m%d-%H%M%S)"
LOGFILE="$LOGDIR/run-$ts.log"

echo "[run] python -u $PYFILE | tee $LOGFILE"
python -u "$PYFILE" 2>&1 | tee "$LOGFILE"

echo "Finished at: $(date)"