#!/usr/bin/env python3
"""
Quick visual sanity checks for the super-resolution dataset and model.

Outputs:
  * Sample HR/LR image pairs saved as a grid.
  * Optionally adds a prediction column when --model-path is supplied.
  * Intensity histogram comparing HR, LR (upsampled), and optional predictions.
  * Text summary of the model architecture for the requested scale.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "code"))
if str(SHARED_ROOT) not in sys.path:
    sys.path.append(str(SHARED_ROOT))

from dataset_paths import (  # noqa: E402
    HR_TRAIN_DIR,
    HR_VALID_DIR,
    LR_TRAIN_DIR,
    LR_VALID_DIR,
    VISUAL_ROOT,
)
from train_adaptive_unet import build_super_resolution_unet  # noqa: E402


def sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z._-]+", "_", value.strip())
    return sanitized.strip("_") or "unnamed"


def sorted_alphanumeric(names: Iterable[str]) -> List[str]:
    def convert(token: str):
        return int(token) if token.isdigit() else token.lower()

    def key(name: str):
        return [convert(part) for part in re.split(r"([0-9]+)", name)]

    return sorted(names, key=key)


def match_lr_filename(hr_filename: str) -> str:
    stem, ext = os.path.splitext(hr_filename)
    return f"{stem}x4{ext}"


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def pick_sample_pairs(
    hr_dir: Path,
    lr_dir: Path,
    num_samples: int,
    seed: int,
) -> List[Tuple[Path, Path]]:
    hr_files = sorted_alphanumeric([name for name in os.listdir(hr_dir) if name.lower().endswith(".png")])

    if not hr_files:
        raise RuntimeError(f"No PNG files found in {hr_dir}")

    pairs: List[Tuple[Path, Path]] = []
    rng = random.Random(seed)
    candidates = hr_files.copy()
    rng.shuffle(candidates)

    for hr_name in candidates:
        lr_name = match_lr_filename(hr_name)
        hr_path = hr_dir / hr_name
        lr_path = lr_dir / lr_name
        if not lr_path.exists():
            continue
        pairs.append((hr_path, lr_path))
        if len(pairs) >= num_samples:
            break

    if len(pairs) < num_samples:
        raise RuntimeError(
            f"Only found {len(pairs)} matching HR/LR pairs in {hr_dir} and {lr_dir}; "
            f"requested {num_samples}."
        )

    return pairs


def make_sample_grid(
    samples: Sequence[Tuple[Path, Path]],
    image_size: int,
    model: tf.keras.Model | None,
) -> Tuple[plt.Figure, dict]:
    rows = len(samples)
    has_model = model is not None
    cols = 4 if has_model else 3
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    collected = {
        "hr": [],
        "lr_native": [],
        "lr_up": [],
        "pred": [],
        "hr_means": [],
        "lr_means": [],
        "pred_means": [],
    }

    for idx, (hr_path, lr_path) in enumerate(samples):
        hr = load_image(hr_path)
        lr_native = load_image(lr_path)

        hr_display = cv2.resize(hr, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        lr_up = cv2.resize(lr_native, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        collected["hr"].append(hr_display)
        collected["lr_native"].append(lr_native)
        collected["lr_up"].append(lr_up)
        collected["hr_means"].append(float(hr_display.mean()))
        collected["lr_means"].append(float(lr_up.mean()))

        axes[idx, 0].imshow(hr_display)
        axes[idx, 0].set_title(f"HR: {hr_path.name}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(lr_native)
        axes[idx, 1].set_title(f"LR native: {lr_path.name}")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(lr_up)
        axes[idx, 2].set_title("LR upsampled")
        axes[idx, 2].axis("off")

        if has_model:
            lr_input = cv2.resize(lr_native, (image_size, image_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            pred = model.predict(lr_input[None, ...], verbose=0)[0]
            pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
            collected["pred"].append(pred)
            collected["pred_means"].append(float(pred.mean()))

            axes[idx, 3].imshow(pred)
            axes[idx, 3].set_title("Prediction")
            axes[idx, 3].axis("off")

    plt.tight_layout()
    return fig, collected


def make_histogram(
    hr_images: Sequence[np.ndarray],
    lr_images: Sequence[np.ndarray],
    pred_images: Sequence[np.ndarray] | None = None,
) -> plt.Figure:
    hist_bins = np.linspace(0.0, 1.0, 256)
    hr_hist = np.zeros_like(hist_bins[:-1], dtype=np.float32)
    lr_hist = np.zeros_like(hist_bins[:-1], dtype=np.float32)
    pred_hist = np.zeros_like(hist_bins[:-1], dtype=np.float32) if pred_images else None

    for hr_img, lr_img in zip(hr_images, lr_images):
        hr_hist += np.histogram(hr_img.ravel(), bins=hist_bins, range=(0.0, 1.0))[0]
        lr_hist += np.histogram(lr_img.ravel(), bins=hist_bins, range=(0.0, 1.0))[0]

    if pred_hist is not None:
        for pred_img in pred_images:
            pred_hist += np.histogram(pred_img.ravel(), bins=hist_bins, range=(0.0, 1.0))[0]

    hr_hist /= hr_hist.sum()
    lr_hist /= lr_hist.sum()
    if pred_hist is not None and pred_hist.sum() > 0:
        pred_hist /= pred_hist.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    centers = hist_bins[:-1]
    ax.plot(centers, hr_hist, label="HR", color="tab:blue")
    ax.plot(centers, lr_hist, label="LR upsampled", color="tab:orange")
    if pred_hist is not None:
        ax.plot(centers, pred_hist, label="Prediction", color="tab:green")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Normalised frequency")
    title = "Intensity distribution (HR vs upsampled LR"
    if pred_hist is not None:
        title += " vs prediction"
    title += ")"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def load_trained_model(model_path: Path) -> tf.keras.Model:
    from shared.custom_layers import ClipAdd, ResizeByScale, ResizeToMatch  # noqa: E402

    custom_objects = {
        "ClipAdd": ClipAdd,
        "ResizeByScale": ResizeByScale,
        "ResizeToMatch": ResizeToMatch,
    }
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    return model


def write_model_summary(
    scale: float,
    image_size: int,
    depth_override: int | None,
    output_path: Path,
    model_path: Path | None,
) -> Tuple[tf.keras.Model, dict]:
    if model_path:
        model = load_trained_model(model_path)
        info = {"scale": scale, "depth": None, "source": str(model_path)}
    else:
        model, info = build_super_resolution_unet(
            scale=scale,
            input_size=image_size,
            depth_override=depth_override,
        )

    lines: List[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines))
    return model, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise dataset samples and inspect the SR model.")
    parser.add_argument("--split", choices=("train", "valid"), default="train", help="Dataset split to sample.")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of HR/LR pairs to visualise.")
    parser.add_argument("--scale", type=float, default=0.6, help="Super-resolution scale (consistent with training).")
    parser.add_argument("--image-size", type=int, default=256, help="HR image size used for training/eval.")
    parser.add_argument("--depth-override", type=int, default=None, help="Optional explicit depth for the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sample selection.")
    parser.add_argument("--scale-label", type=str, default=None, help="Folder tag for storing outputs (defaults to numeric scale).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory root for figures/logs; defaults to project scale_visualizations.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional trained model (.keras) for running predictions.")
    return parser.parse_args()


def main():
    args = parse_args()

    split_map = {
        "train": (HR_TRAIN_DIR, LR_TRAIN_DIR),
        "valid": (HR_VALID_DIR, LR_VALID_DIR),
    }
    hr_dir, lr_dir = split_map[args.split]

    label = args.scale_label or f"{args.scale:.2f}"
    scale_tag = sanitize_for_filename(label)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = VISUAL_ROOT / f"scale_{scale_tag}" / "data_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Sampling {args.num_samples} pairs from {hr_dir} and {lr_dir}")
    samples = pick_sample_pairs(hr_dir, lr_dir, args.num_samples, args.seed)

    model_path = Path(args.model_path).expanduser() if args.model_path else None
    if model_path and not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("[info] Writing model summary...")
    summary_path = output_dir / f"model_summary_scale{args.scale:.2f}.txt"
    model, info = write_model_summary(args.scale, args.image_size, args.depth_override, summary_path, model_path)

    pred_model = model if model_path else None

    print("[info] Generating sample grid...")
    grid_fig, collected = make_sample_grid(samples, args.image_size, pred_model)
    grid_path = output_dir / f"{args.split}_samples.png"
    grid_fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(grid_fig)

    print("[info] Generating intensity histogram...")
    pred_images = collected["pred"] if collected["pred"] else None
    hist_fig = make_histogram(collected["hr"], collected["lr_up"], pred_images if pred_model else None)
    hist_path = output_dir / f"{args.split}_intensity_hist.png"
    hist_fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(hist_fig)

    stats_lines = [
        f"HR mean intensity:  {np.mean(collected['hr_means']):.4f} ± {np.std(collected['hr_means']):.4f}",
        f"LR mean intensity:  {np.mean(collected['lr_means']):.4f} ± {np.std(collected['lr_means']):.4f}",
    ]
    if pred_images:
        stats_lines.append(
            f"Pred mean intensity: {np.mean(collected['pred_means']):.4f} ± {np.std(collected['pred_means']):.4f}"
        )
    stats_lines.extend(
        [
            f"Samples used: {args.num_samples}",
            f"HR source dir: {hr_dir}",
            f"LR source dir: {lr_dir}",
        ]
    )
    if model_path:
        stats_lines.append(f"Model path: {model_path}")
    stats_path = output_dir / f"{args.split}_stats.txt"
    stats_path.write_text("\n".join(stats_lines))

    print("[done] Artifacts written to:")
    print(f"  {grid_path}")
    print(f"  {hist_path}")
    print(f"  {stats_path}")
    print(f"  {summary_path}")
    depth_info = info.get("depth")
    base_channels = info.get("base_channels")
    source = info.get("source", "generated model")
    print(f"Model name: {model.name}, depth={depth_info}, base_channels={base_channels}, source={source}")


if __name__ == "__main__":
    main()
