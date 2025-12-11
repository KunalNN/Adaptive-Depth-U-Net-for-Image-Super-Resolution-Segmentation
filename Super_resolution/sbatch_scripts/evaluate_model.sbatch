#!/bin/bash
#SBATCH --job-name=eval_sr
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
module load anaconda3/2023.03
source activate tensorflow_env

# Variables passed from the submission script
# MODEL_PATH
# TEST_DATA_DIR
# SCALE
# OUTPUT_CSV

echo "Starting evaluation..."
echo "Model: $MODEL_PATH"
echo "Test Data: $TEST_DATA_DIR"
echo "Scale: $SCALE"
echo "Output: $OUTPUT_CSV"

python3 /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/code/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --test_data_dir "$TEST_DATA_DIR" \
    --scale "$SCALE" \
    --output_csv "$OUTPUT_CSV"

echo "Evaluation complete."
