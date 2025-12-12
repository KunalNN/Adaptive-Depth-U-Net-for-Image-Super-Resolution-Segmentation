#!/bin/bash
set -euo pipefail

REPO_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
SR_DIR="$REPO_DIR/Super_resolution"
SBATCH_SCRIPT="$SR_DIR/sbatch_scripts/train_adaptive_simple.sbatch"

# Ensure sbatch script exists
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Error: Sbatch script not found at $SBATCH_SCRIPT"
    exit 1
fi

# Define experiments and scales to process
EXPERIMENTS=("Experiment_1" "Experiment_2")
SCALES=("0.20" "0.50" "0.80")

for EXP in "${EXPERIMENTS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Processing $EXP - Scale $SCALE"
        
        # Construct paths
        # Note: Directory naming convention varies slightly?
        # Based on 'find' output:
        # models/Experiment_1/Scale_0.80_Experiment_1
        # models/Experiment_2/Scale_0.80_Experiment_2
        
        RUN_NAME="Scale_${SCALE}_${EXP}"
        MODEL_DIR="$SR_DIR/models/$EXP/$RUN_NAME"
        LOG_DIR="$SR_DIR/final_data/$EXP/$RUN_NAME"
        
        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "Warning: Model directory not found: $MODEL_DIR. Skipping."
            continue
        fi
        
        # Find checkpoint
        # Look for .keras file. Sort by time (newest first) just in case, though usually only one best exists.
        CHECKPOINT=$(find "$MODEL_DIR" -name "*.keras" -print0 | xargs -0 ls -t | head -n 1)
        
        if [[ -z "$CHECKPOINT" ]]; then
            echo "Warning: No checkpoint found in $MODEL_DIR. Skipping."
            continue
        fi
        
        echo "Found checkpoint: $CHECKPOINT"
        
        # Extract depth from filename
        # Filename format: unet_adaptive_scale_new_loss0.80_depth5.keras
        # We want '5'
        if [[ "$CHECKPOINT" =~ depth([0-9]+)\.keras ]]; then
            DEPTH="${BASH_REMATCH[1]}"
            echo "Extracted depth: $DEPTH"
        else
            echo "Error: Could not extract depth from checkpoint filename. Skipping."
            continue
        fi
        
        # Submit Job
        # We use initial_epoch=60 and epochs=60 to skip training loop (as per my fix in train_adaptive_unet.py)
        # We set patch_size=64 to avoid OOM during evaluation
        
        export SCALE="$SCALE"
        export EXPERIMENT_ID="$EXP"
        export RUN_NAME="$RUN_NAME"
        export MODEL_DIR="$MODEL_DIR"
        export LOG_DIR="$LOG_DIR"
        export RESUME_FROM="$CHECKPOINT"
        export INITIAL_EPOCH=60
        export EPOCHS=60
        export PATCH_SIZE=64
        export PATCHES_PER_IMAGE=6
        export BATCH_SIZE=16
        export EXTRA_ARGS="--depth_override $DEPTH --max_depth $DEPTH"
        
        echo "Submitting job for $RUN_NAME..."
        sbatch "$SBATCH_SCRIPT"
        
    done
done

echo "----------------------------------------------------------------"
echo "All jobs submitted."
