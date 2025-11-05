"""
Generate analysis figures from the per-epoch CSV exports.

Works for both fixed-depth (Experiment 1) and adaptive-depth (Experiment 2)
training runs, provided that you first convert the Slurm logs with
`export_log_metrics.py`.

Typical workflow:

    python3 Super_resolution/code/export_log_metrics.py \
      --logs-root Super_resolution/logs/experiment_1 \
      --output-root Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs

    python3 Super_resolution/code/analyse_experiment_metrics.py \
      --csv-root Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs

Swap the paths above for `experiment_2` to analyse the adaptive-depth runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

SCALE_RE = re.compile(r"scale([0-9.]+)")


@dataclass
class RunSummary:
    label: str
    scale: float
    best_epoch: int
    best_val_loss: float
    best_val_psnr: float
    steps_per_epoch: int
    epoch_time_s: float
    ms_per_step: float


def parse_scale(run_name: str) -> float:
    match = SCALE_RE.search(run_name)
    if not match:
        raise ValueError(f"Could not infer scale from run directory name: {run_name}")
    return float(match.group(1))


def read_run_summary(csv_path: Path) -> RunSummary:
    scale = parse_scale(csv_path.parent.name)

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV {csv_path} is empty.")

    # Candidate rows with validation metrics.
    val_rows = [row for row in rows if row.get("val_loss")]
    if val_rows:
        best_row = min(val_rows, key=lambda row: float(row["val_loss"]))
    else:
        best_row = rows[-1]

    def as_float(row: dict, key: str) -> float:
        value = row.get(key)
        if value in (None, ""):
            return math.nan
        try:
            return float(value)
        except ValueError:
            return math.nan

    best_epoch_val = as_float(best_row, "epoch")
    steps_val = as_float(best_row, "steps_total")
    epoch_time_val = as_float(best_row, "duration_s")
    ms_per_step_val = as_float(best_row, "ms_per_step")

    def cast_to_int(value: float) -> int:
        if math.isnan(value):
            return 0
        return int(value)

    return RunSummary(
        label=csv_path.parent.name,
        scale=scale,
        best_epoch=cast_to_int(best_epoch_val),
        best_val_loss=as_float(best_row, "val_loss"),
        best_val_psnr=as_float(best_row, "val_psnr"),
        steps_per_epoch=cast_to_int(steps_val),
        epoch_time_s=epoch_time_val,
        ms_per_step=ms_per_step_val,
    )


def load_summaries(csv_root: Path) -> List[RunSummary]:
    summaries = [
        read_run_summary(csv_path)
        for csv_path in sorted(csv_root.glob("*/epoch_metrics.csv"))
    ]
    if not summaries:
        raise SystemExit(f"No CSV files found under {csv_root}")
    summaries.sort(key=lambda item: item.scale)
    return summaries


def plot_trend(summaries: Sequence[RunSummary], output_dir: Path) -> None:
    scales = [s.scale for s in summaries]
    losses = [s.best_val_loss for s in summaries]
    psnrs = [s.best_val_psnr for s in summaries]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    loss_line, = ax1.plot(
        scales,
        losses,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Best val loss",
    )
    ax1.set_xlabel("Scale factor (LR/HR ratio)")
    ax1.set_ylabel("Best validation loss (Charbonnier)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    psnr_line, = ax2.plot(
        scales,
        psnrs,
        marker="D",
        linewidth=2,
        linestyle="--",
        color="#d62728",
        label="Best val PSNR",
    )
    ax2.set_ylabel("Best validation PSNR (dB)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.legend(
        (loss_line, psnr_line),
        ("Best val loss", "Best val PSNR"),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "trend_loss_psnr.png", dpi=180)
    plt.close(fig)


def plot_training_speed(summaries: Sequence[RunSummary], output_dir: Path) -> None:
    scales = [s.scale for s in summaries]
    best_epochs = [s.best_epoch for s in summaries]
    epoch_times = [s.epoch_time_s for s in summaries]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    epoch_line, = ax1.plot(
        scales,
        best_epochs,
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Epoch of best val loss",
    )
    ax1.set_xlabel("Scale factor")
    ax1.set_ylabel("Best epoch (lower = faster convergence)", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")

    ax2 = ax1.twinx()
    time_line, = ax2.plot(
        scales,
        epoch_times,
        marker="D",
        linewidth=2,
        linestyle="--",
        color="#ff7f0e",
        label="Epoch duration (s)",
    )
    ax2.set_ylabel("Epoch duration at best checkpoint (s)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    fig.legend(
        (epoch_line, time_line),
        ("Epoch of best val loss", "Epoch duration (s)"),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "training_speed.png", dpi=180)
    plt.close(fig)


def plot_training_load(summaries: Sequence[RunSummary], output_dir: Path) -> None:
    scales = [s.scale for s in summaries]
    steps = [s.steps_per_epoch for s in summaries]
    ms_per_step = [s.ms_per_step for s in summaries]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    bars = ax1.bar(
        scales,
        steps,
        width=0.06,
        color="#9467bd",
        alpha=0.75,
        label="Steps per epoch",
    )
    ax1.set_xlabel("Scale factor")
    ax1.set_ylabel("Steps per epoch", color="#9467bd")
    ax1.set_xlim(0.0, 1.0)
    ax1.tick_params(axis="y", labelcolor="#9467bd")

    ax2 = ax1.twinx()
    step_line, = ax2.plot(
        scales,
        ms_per_step,
        marker="o",
        linewidth=2,
        color="#8c564b",
        label="ms per step",
    )
    ax2.set_ylabel("ms per step (at best val epoch)", color="#8c564b")
    ax2.tick_params(axis="y", labelcolor="#8c564b")

    fig.legend(
        (bars, step_line),
        ("Steps per epoch", "ms per step"),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "training_load.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create summary plots from epoch CSV exports.")
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs"),
        help="Directory containing per-run epoch_metrics.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated PNG figures (default: parent of csv-root).",
    )
    args = parser.parse_args()

    csv_root = args.csv_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else csv_root.parent.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(csv_root)
    plot_trend(summaries, output_dir)
    plot_training_speed(summaries, output_dir)
    plot_training_load(summaries, output_dir)
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
