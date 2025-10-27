#!/usr/bin/env python3
"""Offline evaluation for adaptive-depth U-Net checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "eval_reports"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
parent_root = PROJECT_ROOT.parent
if str(parent_root) not in sys.path:
    sys.path.append(str(parent_root))

from dataset_paths import HR_TRAIN_DIR, HR_VALID_DIR  # noqa: E402
from shared.pipeline import make_eval_patch_dataset, sorted_alphanumeric  # noqa: E402
from train_adaptive_unet import (  # noqa: E402
    build_super_resolution_unet,
    rgb_to_luma_bt601,
)


@dataclass
class EvalResults:
    mse_mean: float
    mse_std: float
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    msssim_mean: float
    msssim_std: float
    samples: int


def infer_eval_shave(scale: float, explicit: int | None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    inv_scale = 1.0 / scale if scale > 0 else 0.0
    scale_factor = int(round(inv_scale)) if inv_scale > 0 else 0
    return 2 * scale_factor if scale_factor > 0 else 0


def load_checkpoint_model(
    model_path: Path,
    scale: float,
    hr_size: int,
    depth_override: int | None,
) -> tf.keras.Model:
    from shared.custom_layers import ClipAdd, ResizeByScale, ResizeToMatch  # noqa: E402

    custom_objects = {
        "ClipAdd": ClipAdd,
        "ResizeByScale": ResizeByScale,
        "ResizeToMatch": ResizeToMatch,
    }
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except ValueError as exc:
        if "Layer node index out of bounds" not in str(exc):
            raise
        print(
            "[warn] load_model failed due to stale serialized graph; rebuilding "
            "the architecture and loading weights instead."
        )
        model, _ = build_super_resolution_unet(
            scale=scale,
            input_size=hr_size,
            depth_override=depth_override,
        )
        model.load_weights(model_path)
        return model


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    eval_shave: int,
) -> Tuple[EvalResults, List[Dict[str, float]]]:
    psnr_vals: List[np.ndarray] = []
    ssim_vals: List[np.ndarray] = []
    msssim_vals: List[np.ndarray] = []
    mse_vals: List[np.ndarray] = []
    per_image: List[Dict[str, float]] = []

    offset = 0
    for lr_batch, hr_batch in dataset:
        pred_rgb = model(lr_batch, training=False)
        pred_rgb = tf.cast(tf.clip_by_value(pred_rgb, 0.0, 1.0), tf.float32)
        hr_rgb = tf.cast(hr_batch, tf.float32)

        pred_y = rgb_to_luma_bt601(pred_rgb)
        hr_y = rgb_to_luma_bt601(hr_rgb)

        if eval_shave > 0:
            pred_y = pred_y[:, eval_shave:-eval_shave, eval_shave:-eval_shave, :]
            hr_y = hr_y[:, eval_shave:-eval_shave, eval_shave:-eval_shave, :]

        batch_psnr = tf.image.psnr(hr_y, pred_y, max_val=1.0).numpy()
        batch_ssim = tf.image.ssim(hr_y, pred_y, max_val=1.0).numpy()
        batch_msssim = tf.image.ssim_multiscale(hr_y, pred_y, max_val=1.0).numpy()
        batch_mse = tf.reduce_mean(tf.square(hr_y - pred_y), axis=[1, 2, 3]).numpy()

        psnr_vals.append(batch_psnr)
        ssim_vals.append(batch_ssim)
        msssim_vals.append(batch_msssim)
        mse_vals.append(batch_mse)

        for i in range(len(batch_psnr)):
            per_image.append(
                {
                    "index": offset + i,
                    "psnr_y": float(batch_psnr[i]),
                    "ssim_y": float(batch_ssim[i]),
                    "msssim_y": float(batch_msssim[i]),
                    "mse_y": float(batch_mse[i]),
                }
            )
        offset += len(batch_psnr)

    if not psnr_vals:
        raise RuntimeError("Evaluation dataset yielded no samples.")

    def stats(values: List[np.ndarray]) -> Tuple[float, float]:
        arr = np.concatenate(values, axis=0).astype(np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    mse_mean, mse_std = stats(mse_vals)
    psnr_mean, psnr_std = stats(psnr_vals)
    ssim_mean, ssim_std = stats(ssim_vals)
    msssim_mean, msssim_std = stats(msssim_vals)

    summary = EvalResults(
        mse_mean=mse_mean,
        mse_std=mse_std,
        psnr_mean=psnr_mean,
        psnr_std=psnr_std,
        ssim_mean=ssim_mean,
        ssim_std=ssim_std,
        msssim_mean=msssim_mean,
        msssim_std=msssim_std,
        samples=len(per_image),
    )
    return summary, per_image


def attach_filenames(per_image: List[Dict[str, float]], filenames: Sequence[str]) -> None:
    if len(per_image) != len(filenames):
        raise ValueError("Per-image metric count does not match filename list.")
    for item, name in zip(per_image, filenames):
        item["filename"] = name


def write_outputs(
    run_dir: Path,
    summary: EvalResults,
    per_image: List[Dict[str, float]],
    config: Dict[str, object],
    write_per_image: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(asdict(summary), indent=2))
    if write_per_image:
        csv_path = run_dir / "per_image_metrics.csv"
        with csv_path.open("w", newline="") as handle:
            fieldnames = ["index", "filename", "psnr_y", "ssim_y", "msssim_y", "mse_y"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_image:
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained adaptive-depth U-Net checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved .keras or .h5 checkpoint.")
    parser.add_argument("--scale", type=float, required=True, help="Downscale factor used during training (0 < scale < 1).")
    parser.add_argument("--hr-dir", type=Path, default=Path(HR_VALID_DIR), help="Directory of high-resolution images to evaluate.")
    parser.add_argument(
        "--lr-dir",
        type=Path,
        default=Path(LR_VALID_DIR),
        help="Directory of low-resolution images. Omit or set to '' to synthesise LR inputs.",
    )
    parser.add_argument("--hr-size", type=int, default=256, help="Image size used during training.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit the number of evaluation samples.")
    parser.add_argument("--eval-shave", type=int, default=None, help="Crop border pixels before metrics (mirrors training logic).")
    parser.add_argument("--depth-override", type=int, default=None, help="Force a specific encoder depth when rebuilding the model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to store evaluation reports.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional folder name inside --output-dir.")
    parser.add_argument("--skip-per-image", action="store_true", help="Do not write per-image CSV metrics.")
    parser.add_argument("--use-train-split", action="store_true", help="Evaluate against the training split defaults instead of validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_train_split:
        if args.hr_dir == Path(HR_VALID_DIR):
            args.hr_dir = Path(HR_TRAIN_DIR)
        if args.lr_dir == Path(LR_VALID_DIR):
            args.lr_dir = Path(LR_TRAIN_DIR)

    hr_dir = Path(args.hr_dir).expanduser()
    if not hr_dir.exists():
        raise FileNotFoundError(f"High-resolution directory not found: {hr_dir}")

    lr_dir = resolve_lr_directory(Path(args.lr_dir).expanduser()) if args.lr_dir else None

    lr_images, hr_images, filenames = load_image_pairs(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        hr_size=args.hr_size,
        scale=args.scale,
        limit=args.limit,
    )

    dataset = build_eval_dataset(lr_images, hr_images, args.batch_size)
    model = load_checkpoint_model(
        model_path=args.model_path.expanduser(),
        scale=args.scale,
        hr_size=args.hr_size,
        depth_override=args.depth_override,
    )

    eval_shave = infer_eval_shave(args.scale, args.eval_shave)
    summary, per_image = evaluate(model, dataset, eval_shave=eval_shave)
    attach_filenames(per_image, filenames)

    print(f"Evaluated {summary.samples} samples.")
    print(f"  PSNR(Y):     {summary.psnr_mean:.4f} ± {summary.psnr_std:.4f} dB")
    print(f"  SSIM(Y):     {summary.ssim_mean:.4f} ± {summary.ssim_std:.4f}")
    print(f"  MS-SSIM(Y):  {summary.msssim_mean:.4f} ± {summary.msssim_std:.4f}")
    print(f"  MSE(Y):      {summary.mse_mean:.6f} ± {summary.mse_std:.6f}")

    output_dir = Path(args.output_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"scale{args.scale:.2f}_{timestamp}"
    run_dir = output_dir / run_name

    config_payload = {
        "model_path": str(args.model_path.expanduser()),
        "scale": args.scale,
        "hr_dir": str(hr_dir),
        "lr_dir": str(lr_dir) if lr_dir else "synthetic",
        "hr_size": args.hr_size,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "eval_shave": eval_shave,
        "depth_override": args.depth_override,
        "samples": summary.samples,
        "created_at": timestamp,
    }

    write_outputs(
        run_dir=run_dir,
        summary=summary,
        per_image=per_image,
        config=config_payload,
        write_per_image=not args.skip_per_image,
    )
    print(f"[done] Report written to {run_dir}")


if __name__ == "__main__":
    main()
