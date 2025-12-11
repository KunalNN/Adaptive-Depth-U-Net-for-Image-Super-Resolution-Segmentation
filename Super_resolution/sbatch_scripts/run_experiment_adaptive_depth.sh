#!/usr/bin/env bash

# Launch Experiment 2: adaptive depth per scale based on the architectural design table.
# Each sbatch submission reuses train_adaptive_simple.sbatch and configures the encoder
# depth to the per-scale target while keeping batch sizes within 2080 Ti limits.

# ------------------------------- Setup -------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"

SCRATCH_ROOT="${SR_SCRATCH_ROOT:-/scratch/knarwani/Final_data/Super_resolution}"
if [[ ! -d "$SCRATCH_ROOT" ]]; then
  echo "[error] Scratch root not found: $SCRATCH_ROOT" >&2
  echo "        Set SR_SCRATCH_ROOT to the desired scratch location before running." >&2
  exit 1
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_EXPERIMENT_ROOT="$REPO_ROOT/experiments/experiment_2_adaptive_depth"
SCRATCH_EXPERIMENT_ROOT="$SCRATCH_ROOT/experiments/experiment_2_adaptive_depth"

# Update output paths to match notebook expectations (final_data)
PROJECT_ROOT="$REPO_ROOT"
LOG_BASE="$PROJECT_ROOT/final_data/Experiment_2"
MODEL_BASE="$PROJECT_ROOT/models/Experiment_2"
EVAL_BASE="$PROJECT_ROOT/final_data/Experiment_2/evaluation"
SUMMARY_ARCHIVE_BASE="$PROJECT_ROOT/final_data/Experiment_2"
META_BASE="$REPO_EXPERIMENT_ROOT/metadata"
EXPERIMENT_ID="Experiment_2"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_BASE" "$MODEL_BASE" "$META_BASE"

# Design table: target depth per scale for Experiment 2 (Adaptive)
SCALES=(
  0.20
  0.50
  0.80
)

declare -A DEPTH_FOR_SCALE=(
  [0.20]=1
  [0.50]=3
  [0.80]=5
)

# Constant batch size for fair comparison
BATCH_SIZE=16

# ------------------------------- Submission Loop -------------------------------

echo "Submitting Experiment 2 runs (Adaptive Depth, Scales: 0.2, 0.5, 0.8)"
for scale in "${SCALES[@]}"; do
  depth="${DEPTH_FOR_SCALE[$scale]:-3}"
  # Notebook expects "Scale_X.XX" in the directory name
  run_name="Scale_${scale}_Experiment_2"
  # Timestamp removed to allow BackupAndRestore to resume from previous run
  # timestamp="$(date +%Y%m%d-%H%M%S)"
  run_suffix="${run_name}"
  log_dir="$LOG_BASE/$run_suffix"
  model_dir="$MODEL_BASE/$run_suffix"
  mkdir -p "$log_dir" "$model_dir"

  export SCALE="$scale"
  export BATCH_SIZE="$BATCH_SIZE"
  export LOG_DIR="$log_dir"
  export MODEL_DIR="$model_dir"
  export RUN_NAME="$run_name"
  # Explicitly set patch size 64, batch size 16, epochs 60, and adaptive depth
  export EXTRA_ARGS="--depth_override ${depth} --max_depth ${depth} --patch_size 64 --epochs 60"
  export EXPERIMENT_ID="$EXPERIMENT_ID"
  export EVAL_OUTPUT_DIR="$EVAL_BASE"
  export SUMMARY_ARCHIVE_DIR="$SUMMARY_ARCHIVE_BASE"

  {
    echo "scale=${scale}"
    echo "batch_size=${BATCH_SIZE}"
    echo "depth=${depth}"
    echo "patch_size=64"
    echo "epochs=60"
    echo "run_name=${run_name}"
    echo "log_dir=${log_dir}"
    echo "model_dir=${model_dir}"
    echo "submitted=$(date --iso-8601=seconds)"
  } > "$META_BASE/${run_suffix}.txt"

  echo "  -> scale=${scale}, depth=${depth}, batch_size=${BATCH_SIZE}, run_name=${run_name}"
  
  # Submit training job and capture ID
  train_job_output=$(sbatch "$SBATCH_SCRIPT")
  train_job_id=$(echo "$train_job_output" | awk '{print $4}')
  echo "     Training Job ID: $train_job_id"

  # Submit evaluation job with dependency
  export MODEL_PATH="$model_dir/best_model.keras" # Assuming best_model.keras is saved
  export TEST_DATA_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/test_data"
  export OUTPUT_CSV="$log_dir/evaluation_summary.csv"
  
  eval_job_output=$(sbatch --dependency=afterok:$train_job_id "$SCRIPT_DIR/evaluate_model.sbatch")
  eval_job_id=$(echo "$eval_job_output" | awk '{print $4}')
  echo "     Evaluation Job ID: $eval_job_id (depends on $train_job_id)"
done

echo "All Experiment 2 jobs submitted. Use 'squeue -u $USER' to monitor them."
