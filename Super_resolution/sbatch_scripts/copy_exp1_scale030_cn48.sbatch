#!/usr/bin/env bash

# Copy all Experiment 1 constant-depth checkpoints from cn48's scratch
# to the persistent repository models directory.

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodelist=cn48
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -J copy-scale030-cn48
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-copy-scale030-cn48-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/slurm-copy-scale030-cn48-%j.out

set -euo pipefail

SRC_ROOT="/scratch/knarwani/Final_data/Super_resolution/experiments/experiment_1_constant_depth_3/models"
DEST_DIR="/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/models/Experiment_1"

echo "[info] Starting copy on $(hostname)"
echo "[info] Source root: $SRC_ROOT"
echo "[info] Destination directory: $DEST_DIR"

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "[error] Source directory not found at $SRC_ROOT" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "[info] Copying all checkpoint files..."
find "$SRC_ROOT" -maxdepth 1 -type f -print0 | xargs -0 -r cp -v -t "$DEST_DIR"

echo "[info] Verifying destination:"
ls -lh "$DEST_DIR"

echo "[info] Copy complete."
