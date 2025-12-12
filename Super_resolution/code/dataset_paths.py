"""
Dataset paths for Super Resolution.
"""
from pathlib import Path

# TODO: USER MUST UPDATE THESE PATHS
# I could not locate the DIV2K dataset. Please update these paths to point to your actual dataset.
# Example: DATA_ROOT = Path("/scratch/knarwani/data/DIV2K")
DATA_ROOT = Path("/scratch/knarwani/Final_data/Super_resolution")

HR_TRAIN_DIR = DATA_ROOT / "DIV2K_train_HR"
LR_TRAIN_DIR = DATA_ROOT / "DIV2K_train_LR_bicubic/X2"

# These should be correct relative to the repo
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs"
