"""
Centralised dataset path definitions for the super-resolution project.

These constants point to the scratch copy of DIV2K used on the cluster so that
training, evaluation, and visualisation scripts stay in sync. Adjust the
`DATA_ROOT` value if the dataset is relocated in the future.
"""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/scratch/knarwani/Final_data/Super_resolution")

# Training splits
HR_TRAIN_DIR = DATA_ROOT / "DIV2K_train_HR"
LR_TRAIN_DIR = DATA_ROOT / "DIV2K_train_LR_bicubic-2" / "X4"

# Validation splits
HR_VALID_DIR = DATA_ROOT / "DIV2K_valid_HR"
LR_VALID_DIR = DATA_ROOT / "DIV2K_valid_LR_bicubic" / "X4"

# Model checkpoints live inside the repo so they survive node-local scratch purges.
MODEL_ROOT = "scratch/knarwani/Final_data/Super_resolution/models"

# TensorBoard logs for training/evaluation runs live in the repo, so they can be
# tailed via TensorBoard while a job runs and persist after node shutdown.
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "tensorboard"

# Local visual inspection artifacts (per-scale plots, grids, etc.).
VISUAL_ROOT = Path(__file__).resolve().parents[1] / "scale_visualizations"
