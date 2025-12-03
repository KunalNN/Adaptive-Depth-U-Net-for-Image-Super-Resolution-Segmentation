#!/usr/bin/env bash

# Resume incomplete experiments for Experiment 1 and Experiment 2
# This script submits jobs starting from the last checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"
EVAL_SBATCH_SCRIPT="$SCRIPT_DIR/evaluate_simple.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$EVAL_SBATCH_SCRIPT" ]]; then
  echo "[error] Expected eval sbatch script not found at $EVAL_SBATCH_SCRIPT" >&2
  exit 1
fi

# Base directories
EXP1_LOG_BASE="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/experiments/experiment_1_constant_depth_3/logs"
EXP1_MODEL_BASE="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/models/Experiment_1"
EXP1_EVAL_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/experiment1/evaluation"

EXP2_LOG_BASE="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/experiment_2"
EXP2_MODEL_BASE="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/models/Experiment_2"
EXP2_EVAL_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/experiment2/evaluation"

# --- Experiment 1 Resumption ---
echo "Submitting Experiment 1 Resumption Jobs..."

# All Exp 1 jobs are complete (>60 epochs) or were not requested to resume.
# Skipping training resumption for Exp 1.

# --- Experiment 2 Resumption ---
echo "Submitting Experiment 2 Resumption Jobs..."

# Scale 0.80 - Incomplete (Epoch ~31)
export SCALE="0.80"
export BATCH_SIZE="1"
export LOG_DIR="$EXP2_LOG_BASE/exp2_adaptive_depth_scale0.80_20251126-150033"
export MODEL_DIR="$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.80_20251126-150033"
export RUN_NAME="exp2_adaptive_depth_scale0.80"
export EXTRA_ARGS="--depth_override 5 --max_depth 5"
export EXPERIMENT_ID="Experiment_2"
export RESUME_FROM="$MODEL_DIR/unet_adaptive_scale_new_loss0.80_depth5.keras"
export INITIAL_EPOCH="41"

echo "  -> Resuming Exp 2 Scale 0.80 from epoch $INITIAL_EPOCH"
sbatch "$SBATCH_SCRIPT"


# --- Evaluation Jobs ---
echo "Submitting Evaluation Jobs..."

# Function to submit evaluation
submit_eval() {
    local scale="$1"
    local model_path="$2"
    local output_dir="$3"
    local run_name="$4"
    local depth_override="$5"

    if [[ ! -f "$model_path" ]]; then
        echo "  [warn] Model not found for $run_name at $model_path, skipping eval."
        return
    fi

    echo "  -> Submitting Eval for $run_name"
    export SCALE="$scale"
    export MODEL_PATH="$model_path"
    export OUTPUT_DIR="$output_dir"
    export RUN_NAME="$run_name"
    export EXTRA_ARGS="--depth_override $depth_override"
    
    sbatch "$EVAL_SBATCH_SCRIPT"
}

# Experiment 1 Evaluations (Constant Depth 3)
# Scales: 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
EXP1_SCALES=("0.20" "0.30" "0.40" "0.50" "0.60" "0.70" "0.80" "0.90")
for scale in "${EXP1_SCALES[@]}"; do
    # Construct model path based on ls output from previous steps
    # Pattern: unet_adaptive_scale_new_loss<SCALE>_depth3.keras
    # Note: Directory timestamp might vary, but we saw them all as 20251126-150711 in the ls output
    model_dir="$EXP1_MODEL_BASE/exp1_depth3_scale${scale}_20251126-150711"
    model_path="$model_dir/unet_adaptive_scale_new_loss${scale}_depth3.keras"
    
    submit_eval "$scale" "$model_path" "$EXP1_EVAL_DIR" "exp1_eval_scale${scale}" "3"
done

# Experiment 2 Evaluations (Adaptive Depth)
# Scales: 0.20 (depth 1), 0.30 (depth 2), 0.40 (depth 3), 0.50 (depth 3), 0.60 (depth 4), 0.70 (depth 5)
# Scale 0.80 is resuming, so we skip eval for now (or it will be eval'd by the training script)

# Scale 0.20 - Depth 1
submit_eval "0.20" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.20_20251126-150033/unet_adaptive_scale_new_loss0.20_depth1.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.20" "1"

# Scale 0.30 - Depth 2
submit_eval "0.30" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.30_20251126-150033/unet_adaptive_scale_new_loss0.30_depth2.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.30" "2"

# Scale 0.40 - Depth 3
submit_eval "0.40" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.40_20251126-150033/unet_adaptive_scale_new_loss0.40_depth3.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.40" "3"

# Scale 0.50 - Depth 3
submit_eval "0.50" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.50_20251126-150033/unet_adaptive_scale_new_loss0.50_depth3.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.50" "3"

# Scale 0.60 - Depth 4
submit_eval "0.60" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.60_20251126-150033/unet_adaptive_scale_new_loss0.60_depth4.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.60" "4"

# Scale 0.70 - Depth 5
submit_eval "0.70" "$EXP2_MODEL_BASE/exp2_adaptive_depth_scale0.70_20251126-150033/unet_adaptive_scale_new_loss0.70_depth5.keras" "$EXP2_EVAL_DIR" "exp2_eval_scale0.70" "5"

echo "All resumption and evaluation jobs submitted."
