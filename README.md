# Super-Resolution

This project trains an adaptive-depth U-Net for single-image super-resolution, with tooling for both workstation experiments and batch jobs on a Slurm GPU cluster. The main training entry point (`code/train_adaptive_unet.py`) reproduces the notebook workflow inside a reusable CLI, handling data degradation, model selection, and evaluation metrics.

## Highlights
- Adaptive encoder depth derived from the requested downscale factor (`infer_depth_from_scale` in `code/train_adaptive_unet.py:117`) keeps the network compact for easier scales.
- Hybrid loss objective mixing pixel, SSIM, and perceptual terms (`code/train_adaptive_unet.py:320`) balances fidelity and perceptual quality.
- Slurm-ready job script (`sbatch_scripts/train_adaptive.sbatch`) that wires up environment modules, GPU policies, and logging.
- Reusable evaluation notebooks in `notebooks/` for qualitative inspection and metric tracking across model variants.

## Repository Layout
```
code/               Training scripts and model builders
dataset/            Local copy of training/validation data
logs/               Console logs from past Slurm jobs
models/             Saved Keras checkpoints and Optuna runs
notebooks/          Jupyter notebooks for experimentation/evaluation
sbatch_scripts/     Slurm job definitions for the RU cluster
requirement.txt     Environment pins for macOS and Linux/GPU setups
```

## Environment Setup
The scripts target Python 3.9–3.11. Create an isolated environment before installing dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt
```

Notes:
- On macOS with Apple silicon, keep the `tensorflow-macos` and `tensorflow-metal` pins.
- On Linux/GPU machines (e.g. the RU cluster), uninstall those macOS wheels first and rely on the CUDA-enabled `tensorflow` and `opencv-python-headless` variants that are already listed in the lower half of `requirement.txt`.

## Data Layout
Training expects a mirrored set of high-resolution and (optionally) low-resolution images:

```
dataset/
├── Raw Data/
│   ├── high_res/   # HR ground truth images
│   └── low_res/    # Optional LR images aligned by filename
├── train/          # pre-split data if you prefer manual loaders
└── val/
```

If `--low_res_dir` is omitted during training, the loader will degrade the HR inputs on-the-fly to create bicubic LR counterparts.

## Local Training
The adaptive script exposes all knobs through CLI arguments. Typical invocation:

```bash
python code/train_adaptive_unet.py \
  --scale 0.6 \
  --high_res_dir "dataset/Raw Data/high_res" \
  --low_res_dir "dataset/Raw Data/low_res" \
  --model_dir models \
  --epochs 100 \
  --batch_size 4
```

Common flags:
- `--mixed_precision` enables the `mixed_float16` policy when a GPU is visible.
- `--limit N` loads only the first `N` HR images (helpful for smoke tests).
- `--depth_override` forces an explicit encoder depth if the scale heuristic needs to be overridden.

Checkpoints are saved as `.keras` artifacts inside the chosen `--model_dir`, while console output and evaluation metrics stream to stdout.

## Slurm Batch Jobs
Use `sbatch_scripts/train_adaptive.sbatch` when running on the RU cluster. Adjust environment variables before submission to point at the correct scratch locations:

```bash
export SCALE=0.6
export HR_DIR="$PWD/dataset/Raw Data/high_res"
export LR_DIR="$PWD/dataset/Raw Data/low_res"
export MODEL_DIR="$PWD/models"
export MIXED_PRECISION=1  # optional
sbatch sbatch_scripts/train_adaptive.sbatch
```

Logs are written to `logs/slurm-*.out`. Each run also mirrors the console stream into a timestamped file under `logs/`.

## Evaluation & Analysis
- Launch Jupyter (`jupyter lab` or `jupyter notebook`) and open any notebook in `notebooks/` for qualitative comparisons and quantitative summaries. Notable entries include `model-eval-adaptive-depth.ipynb` and `u-net-0.6.ipynb`.
- Saved models under `models/` can be loaded directly via `tf.keras.models.load_model`.
- The `models/optuna_run` directory captures hyperparameter search artefacts (Optuna trials).
