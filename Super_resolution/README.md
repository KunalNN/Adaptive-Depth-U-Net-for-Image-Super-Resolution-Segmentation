# Super-Resolution Pipeline

This folder contains the adaptive-depth U-Net training and evaluation stack that we run on the Radboud Science cluster. The code is organised so that data locations, checkpoints, TensorBoard logs, and visual inspection artefacts are all controlled from a single place. This document explains how to prepare the environment, where the data is expected to live, how to launch training (locally or via Slurm), and how to inspect the results.

---

## 1. Dataset Layout and Paths

All scripts import `Super_resolution/code/dataset_paths.py`, which defines the canonical locations of the DIV2K splits and output folders:

```python
DATA_ROOT       = Path("/scratch/knarwani/Final_data/Super_resolution")
HR_TRAIN_DIR    = DATA_ROOT / "DIV2K_train_HR"
LR_TRAIN_DIR    = DATA_ROOT / "DIV2K_train_LR_bicubic-2" / "X4"
HR_VALID_DIR    = DATA_ROOT / "DIV2K_valid_HR"
LR_VALID_DIR    = DATA_ROOT / "DIV2K_valid_LR_bicubic" / "X4"
MODEL_ROOT      = <repo>/Super_resolution/models
LOG_ROOT        = <repo>/Super_resolution/logs/tensorboard
VISUAL_ROOT     = <repo>/Super_resolution/scale_visualizations
```

To point the project at a different dataset copy, edit `DATA_ROOT` once; everything else (training, evaluation, inspection, Slurm jobs) will follow. Expected file layout on disk:

```
/scratch/knarwani/Final_data/Super_resolution/
├── DIV2K_train_HR/*.png
├── DIV2K_train_LR_bicubic-2/X4/*.png
├── DIV2K_valid_HR/*.png
├── DIV2K_valid_LR_bicubic/X4/*.png
├── models/                       # training checkpoints (if you opt to keep them with the dataset)
└── tensorboard/                  # TensorBoard event files (same note as above)
```

Training no longer relies on pre-resized 256×256 frames. All entry points now sample random 256×256 patches from the full-resolution HR images and synthesise the corresponding LR patches on the fly (bicubic down → up). Validation and test runs tile each image into a deterministic grid (stride defaults to the patch size) so metrics remain comparable across scales.

---

## 2. Environment Setup

```bash
cd /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

All tooling (`train_adaptive_unet.py`, TensorBoard, inspection scripts) uses this virtual environment.

---

## 3. Training

### 3.1 Local (interactive) run

```bash
source .venv/bin/activate
python code/train_adaptive_unet.py \
  --scale 0.6 \
  --high_res_dir /scratch/knarwani/Final_data/Super_resolution/DIV2K_train_HR \
  --epochs 200 \
  --patience 20 \
  --batch_size 4 \
  --patch_size 256 \
  --patches_per_image 4 \
  --model_dir /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/models \
  --log_dir /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/tensorboard
```

Key command-line arguments (defaults in parentheses):

| Flag | Description |
|------|-------------|
| `--scale` (**required**) | Down-sampling ratio applied during training (0 < scale < 1). |
| `--high_res_dir` (`HR_TRAIN_DIR`) | Path to the full-resolution DIV2K images. |
| `--low_res_dir` | Ignored in patch mode; LR patches are generated procedurally. |
| `--patch_size` (256) | Edge length of HR/LR patches used for training and evaluation. |
| `--patches_per_image` (4) | Random patches drawn from each training image per epoch (set higher for more coverage). |
| `--eval_stride` (`patch_size`) | Stride for evaluation tiling; lower values increase overlap and patch count. |
| `--shuffle_buffer` (1024) | Shuffle buffer size for the patch dataset. |
| `--preview_patches` (3) | Number of training patches logged to TensorBoard at step 0. |
| `--batch_size` (4) | Batch size for all datasets. |
| `--epochs` (200) | Max epochs before early stopping. |
| `--patience` (20) | Patience for `EarlyStopping`. |
| `--learning_rate` (1e-4) | Adam learning rate. |
| `--val_split` (0.1) / `--test_split` (0.1) | Fractions of the dataset reserved for validation/test. |
| `--base_channels` (64) / `--residual_head_channels` (64) | Width of encoder/decoder blocks. |
| `--depth_override` | Force a specific encoder depth instead of inferring from the scale. |
| `--model_dir` (`MODEL_ROOT`) | Where `.keras` checkpoints are written. |
| `--log_dir` (`LOG_ROOT`) | Where TensorBoard summaries are stored. |
| `--run_name` | Override the auto-generated TensorBoard run name. |
| `--mixed_precision` | Enable `mixed_float16` if GPUs support it. |

Patch sampling is stochastic by default to increase data diversity. Runs become deterministic when you reuse the same `--seed` together with fixed values for `--patches_per_image` and `--eval_stride`, allowing fair comparisons across scale sweeps.

### 3.1.1 Training modes

- **Adaptive depth (default).** `code/train_adaptive_unet.py` infers the encoder/decoder depth from `--scale`. The bundled `sbatch_scripts/train_adaptive.sbatch` wrapper exposes this script for cluster runs; change only the `SCALE` env var to sweep scales while keeping depth adaptive.
- **Fixed depth = 3.** `code/train_adaptive_unet_depth_3.py` forwards to the same trainer but forces `depth_override=3`. Submit via `sbatch_scripts/train_adaptive_depth.sbatch`, e.g.:

  ```bash
  SCALE=0.4 sbatch sbatch_scripts/train_adaptive_depth.sbatch
  ```

  This mode underpins Experiment 1 in the thesis (constant three levels, varying scale).

Both sbatch wrappers set `VAL_SPLIT=TEST_SPLIT=0.05` by default so `val_loss` is available for early stopping and checkpoints.
### 3.2 Loss function

The training objective is a weighted sum of three terms:

```
loss = 1.0 * MSE + 0.1 * (1 - SSIM) + 0.01 * Perceptual(VGG19 block4_conv4)
```

Outputs are clamped to `[0, 1]`. The model also tracks PSNR as the primary metric.

### 3.3 TensorBoard

Run TensorBoard from any node with access to the log directory:

```bash
tensorboard --logdir /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/logs/tensorboard \
            --bind_all --port 6006 --reload_interval 15
```

Forward the port to your laptop (` ssh -L 6006:cn84:6006 knarwani@cnlogin22.science.ru.nl`) and open `http://localhost:6006`. The Scalars tab shows loss/PSNR, Images contains optional samples, and Graphs includes the full network topology.

### 3.4 Cluster (Slurm) workflow

Use the wrapper script to stage data, source the venv, and launch training on a GPU node:

```bash
cd Super_resolution
sbatch sbatch_scripts/train_adaptive.sbatch
```

Useful environment overrides when submitting:

| Variable | Purpose |
|----------|---------|
| `SCALE` | Set the downscale ratio (e.g., `SCALE=0.5`). |
| `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `LEARNING_RATE` | Tune training hyperparameters. |
| `HR_DIR`, `MODEL_DIR`, `LOG_DIR` | Override default paths if you keep data elsewhere. |
| `RUN_NAME` | Friendly label for the TensorBoard run (shows up in the UI). |
| `MIXED_PRECISION=1` | Requests float16 training if GPUs support it. |
| `SYNC_FROM_SHARED=0` | Skip the rsync step if data is already on node-local scratch. |
| `PATCHES_PER_IMAGE` | Override per-image patch sampling when submitting via Slurm. |

Example:

```bash
SCALE=0.6 BATCH_SIZE=8 RUN_NAME=scale06_final sbatch sbatch_scripts/train_adaptive.sbatch
```

Training logs are mirrored to `Super_resolution/logs/` and the compute-node scratch tree. Once the job ends, copy any checkpoints or logs you want to keep off scratch before the scheduled purge.

---

## 4. Visual Inspection Tools

### 4.1 Offline evaluation CLI

`code/evaluate_model.py` loads a saved checkpoint, tiles each validation image into deterministic `patch_size × patch_size` crops, and reports PSNR/SSIM/MSE metrics (plus optional per-patch CSV logs) under `logs/eval_reports/`.

```bash
python code/evaluate_model.py \
  --model-path /home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Super_resolution/models/unet_adaptive_scale_new_loss0.70_depth3.keras \
  --scale 0.6 \
  --patch-size 256 \
  --eval-stride 256 \
  --batch-size 4 \
  --limit 48
```

Outputs:
* `stdout` – aggregate metrics (PSNR/SSIM/MS-SSIM/MSE) and sample count.
* `config.json` – exact inputs and CLI flags used for the run.
* `metrics.json` – summary statistics suitable for dashboards or reports.
* `per_image_metrics.csv` – optional per-patch PSNR/SSIM/MS-SSIM/MSE (omit via `--skip-per-image`).
* All artifacts land in `logs/eval_reports/<run_name>/` (timestamped by default).
* Use `--eval-stride` (default = `patch_size`) to control how densely tiles overlap; keeping the same values across runs ensures reproducible comparisons.

### 4.2 Evaluation notebook as Python

`notebooks/model_eval_0_6.py` is a convert-from-notebook script for headless evaluation. It loads a checkpoint, runs validation samples, and saves per-example grids (HR, LR, prediction, diff heatmaps, Sobel edges).

```bash
python notebooks/model_eval_0_6.py \
  --model-path /scratch/knarwani/Final_data/Super_resolution/models/unet_adaptive_scale0.60_depth3.keras \
  --split valid \
  --scale-label 0.6x \
  --num-examples 10
```

Images are written to `scale_visualizations/scale_0.6x/`. By default the script uses the validation split; switch to `--split train` or supply custom directories with `--high-dir` / `--low-dir`.

---

## 5. Checklist / Troubleshooting

* **“TensorBoard inactive”** – ensure you point `--logdir` at the node-local scratch folder if the job is still running, or copy the run directory to persistent storage before the node reboots.
* **Missing LR files** – training no longer needs them; LR patches are synthesised. Keep the bicubic set only if you want to compare against the official X4 inputs.
* **No GPUs detected** – the trainer disables mixed precision automatically; verify your Slurm allocation includes `--gres=gpu`.
* **Changing data root** – edit `DATA_ROOT` once in `dataset_paths.py`, or set `DATA_ROOT=/path/to/data sbatch ...` when you submit jobs.

For additional questions or edge cases (e.g., running at new scales, integrating new loss terms), leave the existing API calls alone and concentrate changes inside `train_adaptive_unet.py` so the rest of the tooling stays compatible.

---
