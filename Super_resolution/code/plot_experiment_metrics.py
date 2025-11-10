#!/usr/bin/env python3
"""Generate publication-ready plots from experiment evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "experiment_1_constant_depth_3",
        help="Directory that contains the experiment outputs (evaluation/, logs/, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the plots/summary CSV should be written (defaults to <experiment>/plots).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Rendering DPI for the saved figures.",
    )
    return parser.parse_args()


def extract_scale_from_dir(name: str) -> float:
    """Return the numeric scale appearing in a folder name like exp_depth_scale0.50_eval."""
    if "scale" not in name:
        raise ValueError(f"Could not find 'scale' inside folder name: {name}")
    suffix = name.split("scale", maxsplit=1)[-1]
    digits = []
    for ch in suffix:
        if ch.isdigit() or ch == ".":
            digits.append(ch)
        else:
            break
    if not digits:
        raise ValueError(f"Scale digits missing in folder name: {name}")
    return float("".join(digits))


def load_summary_metrics(eval_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing evaluation directory: {eval_dir}")

    for folder in sorted(eval_dir.iterdir()):
        metrics_path = folder / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        scale = extract_scale_from_dir(folder.name)
        metrics["scale"] = scale
        rows.append(metrics)

    if not rows:
        raise RuntimeError(f"No metrics.json files found under {eval_dir}")

    rows.sort(key=lambda item: item["scale"])
    return rows


def load_per_image_metrics(eval_dir: Path, metric_key: str) -> Tuple[List[float], List[str]]:
    values: List[float] = []
    groups: List[str] = []
    for folder in sorted(eval_dir.iterdir()):
        csv_path = folder / "per_image_metrics.csv"
        if not csv_path.exists():
            continue
        scale = extract_scale_from_dir(folder.name)
        with csv_path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if metric_key not in row or row[metric_key] == "":
                    continue
                values.append(float(row[metric_key]))
                groups.append(f"{scale:.2f}")
    return values, groups


def plot_summary_lines(rows: List[Dict[str, float]], output_path: Path, dpi: int) -> None:
    scales = [row["scale"] for row in rows]
    psnr = [row["psnr_mean"] for row in rows]
    psnr_std = [row["psnr_std"] for row in rows]
    ssim = [row["ssim_mean"] for row in rows]
    ssim_std = [row["ssim_std"] for row in rows]
    msssim = [row["msssim_mean"] for row in rows]
    msssim_std = [row["msssim_std"] for row in rows]
    mse = [row["mse_mean"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")

    axes[0].errorbar(scales, psnr, yerr=psnr_std, marker="o", label="PSNR (Y)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Reconstruction fidelity across downscale factors")

    ax_mse = axes[0].twinx()
    ax_mse.plot(scales, mse, color="tab:red", linestyle="--", marker="s", label="MSE (Y)")
    ax_mse.set_ylabel("MSE")
    ax_mse.tick_params(axis="y", labelcolor="tab:red")

    axes[1].errorbar(scales, ssim, yerr=ssim_std, marker="o", label="SSIM (Y)")
    axes[1].errorbar(scales, msssim, yerr=msssim_std, marker="s", label="MS-SSIM (Y)")
    axes[1].set_xlabel("Downscale factor")
    axes[1].set_ylabel("Similarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_mse.get_legend_handles_labels()
    axes[0].legend(handles + handles2, labels + labels2, loc="lower right")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_boxplot(values: List[float], groups: List[str], ylabel: str, output_path: Path, dpi: int) -> None:
    if not values:
        return
    unique_groups = sorted(set(groups), key=lambda item: float(item))
    group_to_vals: Dict[str, List[float]] = {group: [] for group in unique_groups}
    for value, group in zip(values, groups):
        group_to_vals[group].append(value)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.boxplot(
        [group_to_vals[group] for group in unique_groups],
        labels=unique_groups,
        showmeans=True,
        meanline=True,
    )
    ax.set_xlabel("Downscale factor")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} distribution per patch")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    fieldnames = [
        "scale",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "msssim_mean",
        "msssim_std",
        "mse_mean",
        "mse_std",
        "samples",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    eval_dir = experiment_dir / "evaluation"
    output_dir = args.output_dir or (experiment_dir / "plots")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary_metrics(eval_dir)
    write_summary_csv(summary_rows, output_dir / "summary_metrics.csv")
    plot_summary_lines(summary_rows, output_dir / "metrics_vs_scale.png", dpi=args.dpi)

    psnr_values, psnr_groups = load_per_image_metrics(eval_dir, "psnr_y")
    plot_boxplot(psnr_values, psnr_groups, "PSNR (dB)", output_dir / "psnr_boxplot.png", dpi=args.dpi)

    ssim_values, ssim_groups = load_per_image_metrics(eval_dir, "ssim_y")
    plot_boxplot(ssim_values, ssim_groups, "SSIM", output_dir / "ssim_boxplot.png", dpi=args.dpi)

    print(f"Wrote plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
