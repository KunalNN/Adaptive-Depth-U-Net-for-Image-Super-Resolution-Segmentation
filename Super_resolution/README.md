# Super Resolution Pipeline

This folder contains the adaptive-depth U-Net training and evaluation stack for Single Image Super-Resolution.

## 1. Environment Setup

```bash
cd Super_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

## 2. Training

### 2.1 Cluster (Slurm) workflow

Use the wrapper script to launch training on a GPU node:

```bash
cd Super_resolution
sbatch sbatch_scripts/train_adaptive_simple.sbatch
```

**Common Environment Overrides:**

| Variable | Description | Default |
|----------|-------------|---------|
| `SCALE` | Downscaling factor (0 < scale < 1). | `0.5` |
| `DEPTH_OVERRIDE` | Force a specific U-Net depth. | `None` (adaptive) |
| `BATCH_SIZE` | Training batch size. | `8` |
| `EPOCHS` | Number of epochs. | `100` |
| `RUN_NAME` | Name for the run (logs/checkpoints). | `None` |

Example:
```bash
SCALE=0.6 BATCH_SIZE=8 RUN_NAME=scale06_run sbatch sbatch_scripts/train_adaptive_simple.sbatch
```

### 2.2 Batch Experiments

Helper scripts are available to run reproducible experiments:

**Experiment 1: Fixed Depth (Sweep Scales)**
Runs training with a fixed depth of 3 across multiple scales.
```bash
bash sbatch_scripts/run_experiment_fixed_depth.sh
```

**Experiment 2: Adaptive Depth**
Runs training where depth scales with the difficulty/resolution.
```bash
bash sbatch_scripts/run_experiment_adaptive_depth.sh
```

## 3. Evaluation

Evaluation is automatically triggered after training.

To run evaluation manually:
```bash
python code/evaluate_model.py \
  --model-path models/<checkpoint>.keras \
  --scale 0.5 \
  --patch-size 256
```
