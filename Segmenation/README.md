# Segmentation Pipeline

This folder contains the adaptive-depth U-Net training and evaluation stack for Semantic Segmentation.

## 1. Environment Setup

The segmentation pipeline shares the same environment as the Super Resolution pipeline.

```bash
cd ../Super_resolution
source .venv/bin/activate
```

## 2. Training

### 2.1 Cluster (Slurm) workflow

Use the wrapper script to launch training on a GPU node:

```bash
cd Segmenation
sbatch scripts/train_adaptive_simple.sbatch
```

**Common Environment Overrides:**

| Variable | Description | Default |
|----------|-------------|---------|
| `PROTOCOL` | Training protocol (A=ISIC 2017). | `A` |
| `DEPTH` | Encoder depth. | `4` |
| `BATCH_SIZE` | Training batch size. | `0` (auto) |
| `EPOCHS` | Number of epochs. | `0` (auto) |
| `RUN_NAME` | Name for the run (logs/checkpoints). | `unet` |

Example:
```bash
DEPTH=5 BATCH_SIZE=8 RUN_NAME=depth5_run sbatch scripts/train_adaptive_simple.sbatch
```

### 2.2 Batch Experiments

Helper scripts are available to run reproducible experiments:

**Experiment 1: Fixed Depth (Sweep Scales)**
Runs training with a fixed depth of 4 across multiple scales (simulated by dataset or logic).
```bash
bash scripts/run_experiment_fixed_depth.sh
```

**Experiment 2: Adaptive Depth**
Runs training where depth scales with the difficulty/resolution.
```bash
bash scripts/run_experiment_adaptive_depth.sh
```

## 3. Evaluation

Evaluation is automatically triggered after training if `RUN_EVAL_AFTER_TRAIN` is set (default behavior in scripts).

To run evaluation manually:
```bash
python code/evaluate_adaptive_unet.py \
  --checkpoint models/<run_name>.weights.h5 \
  --config logs/<run_name>/config.json
```
