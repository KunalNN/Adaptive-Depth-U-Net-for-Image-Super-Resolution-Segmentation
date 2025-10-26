"""
Centralised dataset path definitions for the ISIC-2017 segmentation project.

These constants point to the scratch copy living under /scratch so that training,
evaluation, and visualisation scripts stay in sync. Adjust `DATA_ROOT` if the
dataset is relocated in the future.
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/scratch/knarwani/Final_data/Segmenation/ISIC-2017")

# Training splits
TRAIN_IMAGE_DIR = DATA_ROOT / "ISIC-2017_Training_Data"
TRAIN_MASK_DIR = DATA_ROOT / "ISIC-2017_Training_Part1_GroundTruth"

# Validation splits
VALID_IMAGE_DIR = DATA_ROOT / "ISIC-2017_Validation_Data"
VALID_MASK_DIR = DATA_ROOT / "ISIC-2017_Validation_Part1_GroundTruth"

# Test splits
TEST_IMAGE_DIR = DATA_ROOT / "ISIC-2017_Test_v2_Data"
TEST_MASK_DIR = DATA_ROOT / "ISIC-2017_Test_v2_Part1_GroundTruth"

# Model checkpoints live inside the repo so they survive node-local scratch purges.
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"

# TensorBoard logs for training/evaluation runs live in the repo, so they can be
# tailed via TensorBoard while a job runs and persist after node shutdown.
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "tensorboard"

# Local visual inspection artifacts (per-scale plots, grids, etc.).
VISUAL_ROOT = Path(__file__).resolve().parents[1] / "scale_visualizations"
