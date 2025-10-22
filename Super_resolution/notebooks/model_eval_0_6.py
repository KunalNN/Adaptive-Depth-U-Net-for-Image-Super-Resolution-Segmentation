#!/usr/bin/env python3
"""
Headless evaluation and visualization script converted from model-eval-0-6.ipynb.

Usage:
    python model_eval_0_6.py \
        --model-path ../../models/best_by_val_loss_adaptive_0.6.keras \
        --dataset-root /scratch/knarwani/Final_data/Super_resolution \
        --output-dir ../../scale_visualizations \
        --scale-label 0.6x

This script mirrors the notebook workflow but saves all figures to disk so it can
run on clusters without interactive display support.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")  # cluster safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.saving import register_keras_serializable
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from dataset_paths import (  # noqa: E402
    DATA_ROOT,
    HR_TRAIN_DIR,
    HR_VALID_DIR,
    LR_TRAIN_DIR,
    LR_VALID_DIR,
    VISUAL_ROOT,
)


def sorted_alphanumeric(names: Iterable[str]) -> List[str]:
    """Sort filenames in a human-friendly way (e.g. 1, 2, 10 instead of 1, 10, 2)."""

    def convert(token: str):
        return int(token) if token.isdigit() else token.lower()

    def key(name: str):
        return [convert(chunk) for chunk in re.split(r"([0-9]+)", name)]

    return sorted(names, key=key)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z._-]+", "_", value.strip())
    return sanitized.strip("_") or "unnamed"


def load_image_directory(directory: Path, size: int, stop_at: str | None = None) -> List[np.ndarray]:
    files = sorted_alphanumeric(os.listdir(directory))
    images: List[np.ndarray] = []

    for name in tqdm(files, desc=f"Loading {directory.name}"):
        if stop_at and name == stop_at:
            break
        img = cv2.imread(str(directory / name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {directory / name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype("float32") / 255.0
        images.append(img_to_array(img))

    if not images:
        raise RuntimeError(f"No images loaded from {directory}")

    return images


def slice_block(data: Sequence[np.ndarray], start: int, end: int | None) -> np.ndarray:
    n = len(data)
    start = max(0, min(start, n))
    end = n if end is None else max(start, min(end, n))
    block = data[start:end]
    if not block:
        raise ValueError(f"Requested slice [{start}:{end}] is empty (dataset has {n} samples).")
    return np.stack(block, axis=0)


@register_keras_serializable(package="resize")
class ResizeByScale(L.Layer):
    def __init__(self, scale, method="bilinear", antialias=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.method = method
        self.antialias = antialias

    def call(self, x):
        x_dtype = x.dtype
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        nh = tf.cast(tf.round(tf.cast(h, tf.float32) * self.scale), tf.int32)
        nw = tf.cast(tf.round(tf.cast(w, tf.float32) * self.scale), tf.int32)
        y = tf.image.resize(tf.cast(x, tf.float32), [nh, nw], method=self.method, antialias=self.antialias)
        return tf.cast(y, x_dtype)

    def get_config(self):
        return {
            "scale": self.scale,
            "method": self.method,
            "antialias": self.antialias,
            **super().get_config(),
        }


@register_keras_serializable(package="resize")
class ResizeToMatch(L.Layer):
    def __init__(self, method="bilinear", antialias=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.antialias = antialias

    def call(self, inputs):
        x, ref = inputs
        target = tf.shape(ref)[1:3]
        y = tf.image.resize(tf.cast(x, tf.float32), target, method=self.method, antialias=self.antialias)
        return tf.cast(y, x.dtype)

    def get_config(self):
        return {
            "method": self.method,
            "antialias": self.antialias,
            **super().get_config(),
        }


@register_keras_serializable(package="utils")
class ClipAdd(L.Layer):
    def call(self, inputs):
        inp, residual = inputs
        y = tf.cast(inp, tf.float32) + tf.cast(residual, tf.float32)
        return tf.cast(tf.clip_by_value(y, 0.0, 1.0), inp.dtype)


def discover_model_path(models_root: Path) -> Path:
    candidates = list(models_root.rglob("*.keras")) + list(models_root.rglob("*.h5"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No *.keras or *.h5 files found under {models_root}. Provide --model-path explicitly."
        )
    raise RuntimeError(
        f"Multiple model files found under {models_root}: {', '.join(str(p) for p in candidates)}. "
        "Please choose one with --model-path."
    )


def load_model(model_path: Path) -> tf.keras.Model:
    custom_objects = {
        "ResizeByScale": ResizeByScale,
        "ResizeToMatch": ResizeToMatch,
        "ClipAdd": ClipAdd,
    }
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"Loaded model '{model.name}' from {model_path}")
    return model


def make_eval_ds(lr_np: np.ndarray, hr_np: np.ndarray, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((lr_np, hr_np))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_dataset(
    ds: tf.data.Dataset, model: tf.keras.Model
) -> Tuple[int, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    all_psnr: List[np.ndarray] = []
    all_ssim: List[np.ndarray] = []
    all_msssim: List[np.ndarray] = []
    n_images = 0

    for lr_b, hr_b in ds:
        pred_b = model(lr_b, training=False)

        if pred_b.shape[1:3] != hr_b.shape[1:3]:
            pred_b = tf.image.resize(pred_b, size=hr_b.shape[1:3], method="bicubic")

        hr_tf = tf.cast(hr_b, tf.float32)
        pred_tf = tf.cast(tf.clip_by_value(pred_b, 0.0, 1.0), tf.float32)

        all_psnr.append(tf.image.psnr(hr_tf, pred_tf, max_val=1.0).numpy())
        all_ssim.append(tf.image.ssim(hr_tf, pred_tf, max_val=1.0).numpy())
        all_msssim.append(tf.image.ssim_multiscale(hr_tf, pred_tf, max_val=1.0).numpy())
        n_images += int(hr_b.shape[0])

    def mean_std(stack: List[np.ndarray]) -> Tuple[float, float]:
        values = np.concatenate(stack, axis=0).astype(np.float64)
        return float(values.mean()), float(values.std())

    m_psnr, s_psnr = mean_std(all_psnr)
    m_ssim, s_ssim = mean_std(all_ssim)
    m_msssim, s_msssim = mean_std(all_msssim)

    return n_images, (m_psnr, s_psnr), (m_ssim, s_ssim), (m_msssim, s_msssim)


def to_gray_u8(x: np.ndarray) -> np.ndarray:
    g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_val = float(np.min(x))
    max_val = float(np.max(x))
    return (x - min_val) / (max_val - min_val + 1e-8)


def crop_around(center: Tuple[int, int], size: int, H: int, W: int) -> Tuple[slice, slice]:
    cy, cx = center
    half = size // 2
    y1 = max(0, cy - half)
    y2 = min(H, cy + half)
    x1 = max(0, cx - half)
    x2 = min(W, cx + half)
    return slice(y1, y2), slice(x1, x2)


def visualize_example(
    model: tf.keras.Model,
    lr_np: np.ndarray,
    hr_np: np.ndarray,
    index: int,
    output_dir: Path,
    scale_label: str,
) -> Path:
    lr_i = lr_np[index].astype(np.float32)
    hr_i = hr_np[index].astype(np.float32)

    pred_i = model.predict(lr_i[None, ...], verbose=0)[0].astype(np.float32)

    H, W = hr_i.shape[:2]

    if pred_i.shape[:2] != (H, W):
        pred_i = cv2.resize(pred_i, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    if lr_i.shape[:2] != (H, W):
        lr_up = cv2.resize(lr_i, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    else:
        lr_up = lr_i

    hr_i = np.clip(hr_i, 0.0, 1.0).astype(np.float32)
    lr_up = np.clip(lr_up, 0.0, 1.0).astype(np.float32)
    pred_i = np.clip(pred_i, 0.0, 1.0).astype(np.float32)

    diff = np.abs(hr_i - pred_i).astype(np.float32)
    diff_gray = diff.mean(axis=2).astype(np.float32)

    maxy, maxx = np.unravel_index(np.argmax(diff_gray), diff_gray.shape)
    crop_size = min(128, min(H, W))
    ys, xs = crop_around((maxy, maxx), crop_size, H, W)

    hr_crop = hr_i[ys, xs].astype(np.float32)
    pred_crop = pred_i[ys, xs].astype(np.float32)
    diff_crop = diff_gray[ys, xs].astype(np.float32)

    h_lr, w_lr = lr_i.shape[:2]
    sy = h_lr / float(H)
    sx = w_lr / float(W)

    lr_y1 = int(round(ys.start * sy))
    lr_y2 = int(round(ys.stop * sy))
    lr_x1 = int(round(xs.start * sx))
    lr_x2 = int(round(xs.stop * sx))

    lr_y1 = np.clip(lr_y1, 0, max(0, h_lr - 1))
    lr_y2 = np.clip(max(lr_y2, lr_y1 + 1), 1, h_lr)
    lr_x1 = np.clip(lr_x1, 0, max(0, w_lr - 1))
    lr_x2 = np.clip(max(lr_x2, lr_x1 + 1), 1, w_lr)

    lr_native_crop = lr_i[lr_y1:lr_y2, lr_x1:lr_x2].astype(np.float32)
    lr_native_crop_up = cv2.resize(
        lr_native_crop,
        (hr_crop.shape[1], hr_crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    lr_up_crop = lr_up[ys, xs].astype(np.float32)

    hr_edges = norm01(
        np.hypot(
            cv2.Sobel(to_gray_u8(hr_i), cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(to_gray_u8(hr_i), cv2.CV_32F, 0, 1, ksize=3),
        )
    )
    pred_edges = norm01(
        np.hypot(
            cv2.Sobel(to_gray_u8(pred_i), cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(to_gray_u8(pred_i), cv2.CV_32F, 0, 1, ksize=3),
        )
    )
    edge_diff = norm01(np.abs(hr_edges - pred_edges))

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    axs[0, 0].imshow(hr_i)
    axs[0, 0].set_title("HR")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(lr_up)
    axs[0, 1].set_title("LR (upsampled)")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(pred_i)
    axs[0, 2].set_title("Prediction")
    axs[0, 2].axis("off")

    im1 = axs[0, 3].imshow(diff_gray, cmap="hot")
    axs[0, 3].set_title("Abs Diff (HR − Pred)")
    axs[0, 3].axis("off")

    axs[0, 4].imshow(edge_diff, cmap="gray")
    axs[0, 4].set_title("Edge Difference (Sobel)")
    axs[0, 4].axis("off")

    axs[1, 0].imshow(hr_crop)
    axs[1, 0].set_title("HR (zoom)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(lr_native_crop_up)
    axs[1, 1].set_title("LR native→up (zoom)")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(lr_up_crop)
    axs[1, 2].set_title("LR-up (zoom)")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(pred_crop)
    axs[1, 3].set_title("Prediction (zoom)")
    axs[1, 3].axis("off")

    im2 = axs[1, 4].imshow(diff_crop, cmap="hot")
    axs[1, 4].set_title("Diff (zoom)")
    axs[1, 4].axis("off")

    fig.colorbar(im1, ax=axs[0, 3], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axs[1, 4], fraction=0.046, pad=0.04)
    plt.tight_layout()

    scale_tag = sanitize_for_filename(scale_label)
    output_file = output_dir / f"scale_{scale_tag}_example_{index:03d}.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_file


def display_n_examples(
    model: tf.keras.Model,
    lr_np: np.ndarray,
    hr_np: np.ndarray,
    output_dir: Path,
    scale_label: str,
    start_index: int,
    n: int,
) -> List[Path]:
    end = min(start_index + n, len(lr_np))
    saved: List[Path] = []
    for idx in range(start_index, end):
        print(f"\n=== Generating visualization for example {idx} ===")
        saved.append(visualize_example(model, lr_np, hr_np, idx, output_dir, scale_label))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless evaluation + visualization for adaptive SR model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the saved Keras model. If omitted, will search under --models-root.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="Root directory to search for a model when --model-path is not supplied.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing the DIV2K splits (defaults to scratch copy).",
    )
    parser.add_argument(
        "--high-dir",
        type=Path,
        help="Directory with high-resolution PNGs. Overrides --dataset-root.",
    )
    parser.add_argument(
        "--low-dir",
        type=Path,
        help="Directory with low-resolution PNGs. Overrides --dataset-root.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default="valid",
        help="Dataset split to use when --high-dir/--low-dir are not provided.",
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        default=None,
        help="Optional filename at which to stop loading (exclusive).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Images are resized to this square size.",
    )
    parser.add_argument(
        "--validation-range",
        type=str,
        default="0:",
        help="Slice for validation data in Python slicing form start:end (default uses entire split).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUAL_ROOT,
        help="Root directory for visualizations (scale-specific subfolders are created automatically).",
    )
    parser.add_argument(
        "--scale-label",
        type=str,
        default="0.6x",
        help="Label used in output filenames to identify the scale.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for visualization.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="How many samples to visualize.",
    )
    return parser.parse_args()


def resolve_dataset_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.high_dir or args.low_dir:
        if not (args.high_dir and args.low_dir):
            raise ValueError("Provide both --high-dir and --low-dir when overriding defaults.")
        return args.high_dir, args.low_dir

    split = args.split

    relative_map = {
        "train": (Path("DIV2K_train_HR"), Path("DIV2K_train_LR_bicubic-2") / "X4"),
        "valid": (Path("DIV2K_valid_HR"), Path("DIV2K_valid_LR_bicubic") / "X4"),
    }

    default_map = {
        "train": (HR_TRAIN_DIR, LR_TRAIN_DIR),
        "valid": (HR_VALID_DIR, LR_VALID_DIR),
    }

    base = args.dataset_root
    if base == DATA_ROOT:
        high_dir, low_dir = default_map[split]
    else:
        rel_hr, rel_lr = relative_map[split]
        high_dir = base / rel_hr
        low_dir = base / rel_lr

    if not high_dir.exists():
        raise FileNotFoundError(f"High-resolution directory not found: {high_dir}")
    if not low_dir.exists():
        raise FileNotFoundError(f"Low-resolution directory not found: {low_dir}")
    return high_dir, low_dir


def parse_slice(slice_text: str) -> Tuple[int, int | None]:
    if ":" not in slice_text:
        raise ValueError("--validation-range must be in start:end form")
    start_str, end_str = slice_text.split(":", 1)
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else None
    return start, end


def main():
    args = parse_args()

    high_dir, low_dir = resolve_dataset_dirs(args)
    print(f"High-res images from: {high_dir}")
    print(f"Low-res images from : {low_dir}")

    high_images = load_image_directory(high_dir, size=args.image_size, stop_at=args.stop_at)
    low_images = load_image_directory(low_dir, size=args.image_size, stop_at=args.stop_at)

    if len(high_images) != len(low_images):
        raise RuntimeError(
            f"Mismatch between high-res ({len(high_images)}) and low-res ({len(low_images)}) counts."
        )

    v_start, v_end = parse_slice(args.validation_range)
    val_high = slice_block(high_images, v_start, v_end)
    val_low = slice_block(low_images, v_start, v_end)

    val_ds = make_eval_ds(val_low, val_high, batch_size=args.batch_size)

    model_path = args.model_path or discover_model_path(args.models_root)
    model = load_model(model_path)

    n_val, (mP, sP), (mS, sS), (mMS, sMS) = evaluate_dataset(val_ds, model)
    print(f"[Validation] images: {n_val}")
    print(f"  PSNR    : {mP:.4f} ± {sP:.4f} dB")
    print(f"  SSIM    : {mS:.4f} ± {sS:.4f}")
    print(f"  MS-SSIM : {mMS:.4f} ± {sMS:.4f}")

    output_root = Path(args.output_dir)
    scale_tag = sanitize_for_filename(args.scale_label)
    if output_root.name != f"scale_{scale_tag}":
        output_dir = ensure_dir(output_root / f"scale_{scale_tag}")
    else:
        output_dir = ensure_dir(output_root)
    print(f"Saving visualizations to: {output_dir}")

    saved_paths = display_n_examples(
        model,
        val_low,
        val_high,
        output_dir,
        scale_label=args.scale_label,
        start_index=args.start_index,
        n=args.num_examples,
    )

    print("\nFinished generating visualizations:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
