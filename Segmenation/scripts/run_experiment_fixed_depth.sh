#!/usr/bin/env bash

# Launch Experiment 1: fixed depth (3 levels) while sweeping the requested
# reconstruction scale. Each sbatch submission reuses train_adaptive_simple.sbatch
# and pins the encoder depth to three levels by passing --depth_override 3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"

SCRATCH_ROOT="${SR_SCRATCH_ROOT:-/scratch/knarwani/Final_data/Super_resolution}"
if [[ ! -d "$SCRATCH_ROOT" ]]; then
  echo "[error] Scratch root not found: $SCRATCH_ROOT" >&2
  echo "        Set SR_SCRATCH_ROOT to the desired scratch location before running." >&2
  exit 1
fi

REPO_EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/experiments/experiment_1_constant_depth_3"
SCRATCH_EXPERIMENT_ROOT="$SCRATCH_ROOT/experiments/experiment_1_constant_depth_3"

LOG_BASE="$REPO_EXPERIMENT_ROOT/logs"
MODEL_BASE="$(cd "$SCRIPT_DIR/.." && pwd)/models/Experiment_1"
META_BASE="$REPO_EXPERIMENT_ROOT/metadata"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_BASE" "$MODEL_BASE" "$META_BASE"

# Scale sweep defined in Table 1 (Experiment 1).
SCALES=(
  0.20
  0.30
  0.40
  0.50
  0.60
  0.70
  0.80
  0.90
)

# Batch sizes chosen conservatively so the largest scales fit on a 2080 Ti.
declare -A BATCH_SIZE_FOR_SCALE=(
  [0.20]=8
  [0.30]=8
  [0.40]=8
  [0.50]=6
  [0.60]=4
  [0.70]=2
  [0.80]=1
  [0.90]=1
)

echo "Submitting Experiment 1 runs (depth=3, varying scale)"
for scale in "${SCALES[@]}"; do
  batch_size="${BATCH_SIZE_FOR_SCALE[$scale]:-4}"
  run_name="exp1_depth3_scale${scale}"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_suffix="${run_name}_${timestamp}"
  log_dir="$LOG_BASE/$run_suffix"
  model_dir="$MODEL_BASE/$run_suffix"
  mkdir -p "$log_dir" "$model_dir"

  export SCALE="$scale"
  export BATCH_SIZE="$batch_size"
  export LOG_DIR="$log_dir"
  export MODEL_DIR="$model_dir"
  export RUN_NAME="$run_name"
  export EXTRA_ARGS="--depth_override 3"


  {
    echo "scale=${scale}"
    echo "batch_size=${batch_size}"
    echo "depth=3"
    echo "run_name=${run_name}"
    echo "log_dir=${log_dir}"
    echo "model_dir=${model_dir}"
    echo "submitted=$(date --iso-8601=seconds)"
  } > "$META_BASE/${run_suffix}.txt"

  echo "  -> scale=${scale}, batch_size=${batch_size}, run_name=${run_name}, log_dir=${log_dir}, model_dir=${model_dir}"
  sbatch "$SBATCH_SCRIPT"
done

echo "All Experiment 1 jobs submitted. Use 'squeue -u $USER' to monitor them."
