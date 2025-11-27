#!/usr/bin/env bash

# Launch Experiment 2: adaptive depth per scale based on the architectural design table.
# Each sbatch submission reuses train_adaptive_simple.sbatch and configures the encoder
# depth to the per-scale target while keeping batch sizes within 2080 Ti limits.

# ------------------------------- Setup -------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_EXPERIMENT_ROOT="$REPO_ROOT/experiments/experiment_2_adaptive_depth"
RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$REPO_EXPERIMENT_ROOT/runs}"
LOG_BASE="$RUN_ROOT/logs"
MODEL_BASE="$RUN_ROOT/models"
META_BASE="$REPO_EXPERIMENT_ROOT/metadata"
TRAINING_CSV="$REPO_EXPERIMENT_ROOT/training_runs.csv"
EVAL_CSV="$REPO_EXPERIMENT_ROOT/evaluation_runs.csv"
PAIRS_MANIFEST="$REPO_ROOT/manifests/isic2017_train_val_pairs.json"
GLOBAL_EXTRA_ARGS="${GLOBAL_EXTRA_ARGS:-}"
PROTOCOL="${PROTOCOL:-A}"
export PROTOCOL

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_BASE" "$MODEL_BASE" "$META_BASE" "$(dirname "$PAIRS_MANIFEST")"
mkdir -p "$(dirname "$TRAINING_CSV")" "$(dirname "$EVAL_CSV")"

ensure_csv_header() {
  local csv_path="$1"
  local header="$2"
  if [[ ! -f "$csv_path" ]]; then
    echo "$header" > "$csv_path"
  fi
}

ensure_csv_header "$TRAINING_CSV" "submitted_at,job_id,scale,batch_size,depth,run_name,log_dir,model_dir"
ensure_csv_header "$EVAL_CSV" "submitted_at,job_id,run_name,scale,batch_size,depth,log_dir,model_dir,config_path,status"

# ------------------------------- Experiment Setup -------------------------------

# Design table: target depth per scale with conservative batch sizes for a 2080 Ti.
SCALES=(
  0.20
  0.30
  0.40
  0.50
  0.60
  0.70
  0.80
)

# Depth per scale comes straight from the architecture table (ignore intermediate feature sizes).
declare -A DEPTH_FOR_SCALE=(
  [0.20]=1
  [0.30]=2
  [0.40]=3
  [0.50]=3
  [0.60]=4
  [0.70]=5
  [0.80]=5
)

declare -A BATCH_SIZE_FOR_SCALE=(
  [0.20]=8
  [0.30]=8
  [0.40]=8
  [0.50]=6
  [0.60]=4
  [0.70]=2
  [0.80]=1
)

# ------------------------------- Submission Loop -------------------------------

echo "Submitting Experiment 2 runs (adaptive depth per scale)"
for scale in "${SCALES[@]}"; do
  depth="${DEPTH_FOR_SCALE[$scale]:-3}"
  batch_size="${BATCH_SIZE_FOR_SCALE[$scale]:-2}"
  run_name="exp2_adaptive_depth_scale${scale}"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_suffix="${run_name}_${timestamp}"
  scale_root="$RUN_ROOT/scale_${scale}"
  log_root="$scale_root/training/logs"
  model_root="$scale_root/training/models"
  meta_root="$scale_root/metadata"
  log_dir="$log_root/$run_suffix"
  model_dir="$model_root/$run_suffix"

  mkdir -p "$log_root" "$model_root" "$meta_root"

  export SCALE="$scale"
  export BATCH_SIZE="$batch_size"
  export DEPTH="$depth"
  export LOG_DIR="$log_dir"
  export MODEL_DIR="$model_dir"
  export RUN_NAME="$run_name"
  export PAIRS_MANIFEST="$PAIRS_MANIFEST"
  export EXTRA_ARGS="$GLOBAL_EXTRA_ARGS"
  export REPO_DIR="$REPO_ROOT"

  {
    echo "scale=${scale}"
    echo "batch_size=${batch_size}"
    echo "depth=${depth}"
    echo "weight_decay=1e-4"
    echo "patience_strategy=protocol_default_plus5_if_batch_size_le_2"
    echo "run_name=${run_name}"
    echo "log_dir=${log_dir}"
    echo "model_dir=${model_dir}"
    echo "pairs_manifest=${PAIRS_MANIFEST}"
    echo "submitted=$(date --iso-8601=seconds)"
  } | tee "$META_BASE/${run_suffix}.txt" > "$meta_root/${run_suffix}.txt"

  echo "  -> scale=${scale}, depth=${depth}, batch_size=${batch_size}, run_name=${run_name}"
  submit_output="$(sbatch "$SBATCH_SCRIPT")"
  echo "$submit_output"
  job_id="$(awk '{print $4}' <<<"$submit_output")"

  if [[ -n "$job_id" ]]; then
    submission_iso="$(date --iso-8601=seconds)"
    {
      printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$submission_iso" "$job_id" "$scale" "$batch_size" "$DEPTH" "$run_name" "$log_dir" "$model_dir"
    } >> "$TRAINING_CSV"

    config_path="$log_dir/$run_name/config.json"
    {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$submission_iso" "$job_id" "$run_name" "$scale" "$batch_size" "$DEPTH" "$log_dir" "$model_dir" "$config_path" "pending"
    } >> "$EVAL_CSV"
  fi
done

echo "All Experiment 2 jobs submitted. Use 'squeue -u $USER' to monitor them."
