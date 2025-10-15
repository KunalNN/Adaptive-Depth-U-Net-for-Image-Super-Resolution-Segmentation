#!/usr/bin/env python3
"""
Batch runner for Experiment 1 (fixed depth, varying scale).

This script executes `train_adaptive_unet.py` multiple times while keeping the
encoder depth fixed and sweeping over the requested scales. Training logs are
captured per scale and validation/test PSNR & SSIM statistics are aggregated
into a summary CSV for easy reporting.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
TRAIN_SCRIPT = CODE_DIR / "train_adaptive_unet.py"

DEFAULT_SCALES = [0.20, 0.30, 0.40, 0.50, 0.70, 0.90]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 sweeps for adaptive U-Net SR.")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Scale factors to evaluate (default: %(default)s).",
    )
    parser.add_argument("--depth", type=int, default=3, help="Encoder depth override to enforce.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs to train each model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--hr-size", type=int, default=256, help="High-resolution crop size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on dataset size.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed shared across runs.")
    parser.add_argument("--high-res-dir", type=str, default=None, help="Override HR directory.")
    parser.add_argument("--low-res-dir", type=str, default=None, help="Optional LR directory.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "experiment1"),
        help="Root directory for checkpoints (per-scale subfolders will be created).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "experiment1"),
        help="Directory where logs and summary files are written.",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed_float16 policy if GPU is present.")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the training script (supply after '--').",
    )
    return parser.parse_args()


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_training(
    scale: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    log_dir = ensure_path(output_dir / f"scale_{scale:.2f}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    model_dir = ensure_path(Path(args.model_dir) / f"scale_{scale:.2f}")

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--scale",
        f"{scale:.2f}",
        "--depth_override",
        str(args.depth),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hr_size",
        str(args.hr_size),
        "--seed",
        str(args.seed),
        "--model_dir",
        str(model_dir),
        "--image_suffix",
        args.image_suffix,
    ]

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.high_res_dir:
        cmd.extend(["--high_res_dir", args.high_res_dir])
    if args.low_res_dir:
        cmd.extend(["--low_res_dir", args.low_res_dir])
    if args.mixed_precision:
        cmd.append("--mixed_precision")
    if args.extra_args:
        cmd.extend(args.extra_args)

    metrics: Dict[str, Dict[str, float]] = {}
    split_pattern = re.compile(r"^(Validation|Test) samples evaluated")
    psnr_pattern = re.compile(r"^\s+PSNR\s+: ([0-9.]+)\s±\s([0-9.]+)")
    ssim_pattern = re.compile(r"^\s+SSIM\s+: ([0-9.]+)\s±\s([0-9.]+)")
    msssim_pattern = re.compile(r"^\s+MS-SSIM\s+: ([0-9.]+)\s±\s([0-9.]+)")

    current_split: Optional[str] = None
    log_lines: List[str] = []

    print(f"\n=== Running scale {scale:.2f} (depth {args.depth}) ===")
    print("Command:", " ".join(cmd))

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        assert proc.stdout is not None  # for type checkers
        for line in proc.stdout:
            print(line, end="")
            log_lines.append(line)

            split_match = split_pattern.search(line)
            if split_match:
                current_split = split_match.group(1)
                metrics.setdefault(current_split, {})
                continue

            if current_split:
                psnr_match = psnr_pattern.search(line)
                if psnr_match:
                    metrics[current_split]["psnr_mean"] = float(psnr_match.group(1))
                    metrics[current_split]["psnr_std"] = float(psnr_match.group(2))
                    continue

                ssim_match = ssim_pattern.search(line)
                if ssim_match:
                    metrics[current_split]["ssim_mean"] = float(ssim_match.group(1))
                    metrics[current_split]["ssim_std"] = float(ssim_match.group(2))
                    continue

                msssim_match = msssim_pattern.search(line)
                if msssim_match:
                    metrics[current_split]["ms_ssim_mean"] = float(msssim_match.group(1))
                    metrics[current_split]["ms_ssim_std"] = float(msssim_match.group(2))
                    continue

        return_code = proc.wait()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file.write_text("".join(log_lines))

    if return_code != 0:
        raise RuntimeError(f"Training failed for scale {scale:.2f} (exit code {return_code}). See {log_file}.")

    return metrics


def write_summary(output_dir: Path, summary: List[Dict[str, object]]) -> None:
    ensure_path(output_dir)
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "scale",
        "split",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "ms_ssim_mean",
        "ms_ssim_std",
    ]

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"\nSummary written to {csv_path}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_path(Path(args.output_dir))

    summary_rows: List[Dict[str, object]] = []
    for scale in args.scales:
        run_metrics = run_training(scale, args, output_dir)
        for split_name, metrics in run_metrics.items():
            row = {"scale": scale, "split": split_name}
            row.update(metrics)
            summary_rows.append(row)

    write_summary(output_dir, summary_rows)


if __name__ == "__main__":
    main()
