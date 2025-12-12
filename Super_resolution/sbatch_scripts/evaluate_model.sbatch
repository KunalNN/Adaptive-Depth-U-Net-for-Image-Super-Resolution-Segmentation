#!/bin/bash
#SBATCH --job-name=eval_sr
#SBATCH -A cseduproject
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/%x_%j.log
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load environment
source /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/.venv/bin/activate

# Base directories
REPO_ROOT="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation"
MODELS_DIR="$REPO_ROOT/Super_resolution/models/Experiment_2"
FINAL_DATA_DIR="$REPO_ROOT/Super_resolution/final_data/Experiment_2"
TEST_DATA_DIR="$REPO_ROOT/Super_resolution/test_data"

echo "Starting batch evaluation for Experiment 2..."

# Iterate over model directories
for model_dir in "$MODELS_DIR"/*; do
    if [ -d "$model_dir" ]; then
        dir_name=$(basename "$model_dir")
        
        # Extract scale from directory name (e.g., Scale_0.20_Experiment_2)
        # We assume the format Scale_X.XX_Experiment_2
        scale=$(echo "$dir_name" | grep -oP 'Scale_\K[0-9.]+')
        
        if [ -z "$scale" ]; then
            echo "[warn] Could not extract scale from $dir_name, skipping."
            continue
        fi
        
        echo "Processing $dir_name (Scale: $scale)..."
        
        # Find the .keras model file
        # We look for the newest .keras file if multiple exist, or just the one expected
        model_path=$(find "$model_dir" -name "*.keras" | head -n 1)
        
        if [ -z "$model_path" ]; then
            echo "[warn] No .keras model found in $model_dir, skipping."
            continue
        fi
        
        # Construct output CSV path
        # It should go to final_data/Experiment_2/Scale_X.XX_Experiment_2/evaluation.csv
        # Note: The user seems to have a nested structure in final_data based on previous logs:
        # final_data/Experiment_2/Scale_0.20_Experiment_2/Scale_0.20_Experiment_2/evaluation.csv ?
        # Or just final_data/Experiment_2/Scale_0.20_Experiment_2/evaluation.csv
        # Let's check the structure created by training.
        # The training log said: log_dir=.../final_data/Experiment_2/Scale_0.20_Experiment_2
        # And inside that, it created run_name=Scale_0.20_Experiment_2 (inferred or passed)
        # So it might be .../Scale_0.20_Experiment_2/Scale_0.20_Experiment_2/evaluation.csv
        
        # Let's try to find the run directory inside final_data
        target_final_dir="$FINAL_DATA_DIR/$dir_name"
        
        # If the run directory exists inside, use it. Otherwise use the parent.
        # We want to put evaluation.csv where config.json is, usually.
        # Let's search for config.json in the target directory to find the actual run folder
        run_dir=$(find "$target_final_dir" -name "config.json" -printf "%h\n" | head -n 1)
        
        if [ -z "$run_dir" ]; then
            # If no config.json found, maybe just use the target dir itself
            run_dir="$target_final_dir"
            mkdir -p "$run_dir"
        fi
        
        output_csv="$run_dir/evaluation.csv"
        
        if [ -f "$output_csv" ]; then
            echo "[info] Evaluation already exists at $output_csv, skipping."
        else
            echo "Running evaluation for $scale..."
            echo "  Model: $model_path"
            echo "  Output: $output_csv"
            
            python3 "$REPO_ROOT/Super_resolution/code/evaluate_model.py" \
                --model_path "$model_path" \
                --test_data_dir "$TEST_DATA_DIR" \
                --scale "$scale" \
                --output_csv "$output_csv"
                
            echo "  Done."
        fi
    fi
done

echo "Batch evaluation complete."
