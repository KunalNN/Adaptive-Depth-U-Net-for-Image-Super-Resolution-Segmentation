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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"
EVAL_SBATCH_SCRIPT="$SCRIPT_DIR/evaluate_model.sbatch"

# Setup Output Paths
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_BASE="$PROJECT_ROOT/final_data/Test_Run"
MODEL_BASE="$PROJECT_ROOT/models/Test_Run"
mkdir -p "$LOG_BASE" "$MODEL_BASE"

# Test Configuration (Fastest possible run)
SCALE=0.20
DEPTH=1
BATCH_SIZE=16
EPOCHS=1
RUN_NAME="Test_Run_Scale_${SCALE}"

# Setup Run Directory
run_suffix="${RUN_NAME}"
log_dir="$LOG_BASE/$run_suffix"
model_dir="$MODEL_BASE/$run_suffix"
mkdir -p "$log_dir" "$model_dir"

# Export Variables for Training
export SCALE="$SCALE"
export BATCH_SIZE="$BATCH_SIZE"
export LOG_DIR="$log_dir"
export MODEL_DIR="$model_dir"
export RUN_NAME="$RUN_NAME"
export EXTRA_ARGS="--depth_override ${DEPTH} --max_depth ${DEPTH} --patch_size 64 --epochs ${EPOCHS}"
export EXPERIMENT_ID="Test_Run"

echo "Submitting TEST Training Job..."
echo "  -> Scale: $SCALE, Depth: $DEPTH, Epochs: $EPOCHS"

# Submit Training
# train_job_output=$(sbatch "$SBATCH_SCRIPT")
# train_job_id=$(echo "$train_job_output" | awk '{print $4}')
# echo "     Training Job ID: $train_job_id"

# Export Variables for Evaluation
# Find the actual model file (since name includes loss/depth)
export MODEL_PATH=$(ls "$model_dir"/*.keras | head -n 1)
export TEST_DATA_DIR="$PROJECT_ROOT/test_data"
export OUTPUT_CSV="$log_dir/evaluation_summary.csv"

echo "Submitting TEST Evaluation Job..."

# Submit Evaluation (No dependency since training is skipped)
eval_job_output=$(sbatch "$EVAL_SBATCH_SCRIPT")
eval_job_id=$(echo "$eval_job_output" | awk '{print $4}')
echo "     Evaluation Job ID: $eval_job_id"

echo "Test pipeline submitted. Monitor with 'squeue -u $USER'."
