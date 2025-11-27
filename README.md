# Adaptive-Depth U-Net for Image Super-Resolution & Segmentation

This repository contains two deep learning pipelines for adaptive-depth U-Nets.

## 🚀 Quick Start

### 1. Super Resolution
**Goal:** Upscale low-resolution images.
**Detailed Docs:** [Super_resolution/README.md](Super_resolution/README.md)

**Run on Cluster:**
```bash
cd Super_resolution
# Submit a single training job
sbatch sbatch_scripts/train_adaptive_simple.sbatch

# Run full experiments
bash sbatch_scripts/run_experiment_fixed_depth.sh
```

**Key Flags:**
- `SCALE`: Downscaling factor (0.2 - 0.9).
- `DEPTH_OVERRIDE`: Force a specific U-Net depth.

### 2. Segmentation
**Goal:** Semantic segmentation (ISIC 2017 dataset).
**Detailed Docs:** [Segmenation/README.md](Segmenation/README.md)

**Run on Cluster:**
```bash
cd Segmenation
# Submit a single training job
sbatch scripts/train_adaptive_simple.sbatch

# Run full experiments
bash scripts/run_experiment_fixed_depth.sh
```

**Key Flags:**
- `PROTOCOL`: Dataset protocol (default 'A').
- `DEPTH`: Encoder depth (default 4).

## 🛠️ Environment
Both pipelines use the same Python environment located in `Super_resolution/.venv`.
```bash
source Super_resolution/.venv/bin/activate
```
