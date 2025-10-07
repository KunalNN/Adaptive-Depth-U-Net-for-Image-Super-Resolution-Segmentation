#!/usr/bin/env bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio
#SBATCH --qos=csedu-small
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 03:00:00
#SBATCH -J unet-train
#SBATCH -o logs/slurm-%x-%j.out
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
PYFILE="$REPO_DIR/code/u-net-custom-scale-adative-depth.py"
LOGDIR="$REPO_DIR/logs"
MODELDIR="$REPO_DIR/models"


mkdir -p "$LOGDIR" "$MODELDIR"

# ─── Python / Env ──────────────────────────────────────────────────────────────
source "/home/knarwani/thesis/super-resolution/.venv/bin/activate"
# ─── Runtime env tweaks (safe defaults for TF) ─────────────────────────────────
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Threading tuned to the CPUs you requested
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TF_NUM_INTEROP_THREADS=2

# (Optional) Make sure scratch data is reachable; print for debugging.
echo "[data] scratch: /scratch/$USER"
du -sh "/scratch/$USER" 2>/dev/null || true

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