"""
Convert `run-simple-*.log` training transcripts into per-epoch CSV files.

The script is experiment-agnostic: point it at any logs directory populated by
the Slurm jobs (e.g. `Super_resolution/logs/experiment_1` or
`Super_resolution/logs/experiment_2`) and choose where the CSV files should live.

Example usages:

    # Experiment 1 (fixed depth)
    python3 Super_resolution/code/export_log_metrics.py \
      --logs-root Super_resolution/logs/experiment_1 \
      --output-root Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs

    # Experiment 2 (adaptive depth)
    python3 Super_resolution/code/export_log_metrics.py \
      --logs-root Super_resolution/logs/experiment_2 \
      --output-root Super_resolution/experiments/experiment_2_adaptive_depth/csv_logs
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

EPOCH_RE = re.compile(r"^Epoch\s+(\d+)")
PROGRESS_RE = re.compile(r"^(?P<done>\d+)\s*/\s*(?P<total>\d+)$")


def parse_metrics_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a single epoch summary line emitted by Keras."""
    if " - loss:" not in line or "ms/step" not in line:
        return None

    parts = [part.strip() for part in line.strip().split(" - ") if part.strip()]
    if len(parts) < 4:
        return None

    metrics: Dict[str, float] = {}

    progress = parts.pop(0)
    match = PROGRESS_RE.match(progress)
    if not match:
        return None
    metrics["steps_completed"] = float(match.group("done"))
    metrics["steps_total"] = float(match.group("total"))

    duration = parts.pop(0)
    if duration.endswith("s"):
        duration = duration[:-1]
    try:
        metrics["duration_s"] = float(duration)
    except ValueError:
        metrics["duration_s"] = float("nan")

    ms_per_step = parts.pop(0).replace("ms/step", "").strip()
    try:
        metrics["ms_per_step"] = float(ms_per_step)
    except ValueError:
        metrics["ms_per_step"] = float("nan")

    for item in parts:
        if ":" not in item:
            continue
        key, value = (seg.strip() for seg in item.split(":", 1))
        try:
            metrics[key.lower()] = float(value)
        except ValueError:
            continue

    return metrics


def extract_epoch_rows(log_path: Path) -> List[Dict[str, float]]:
    """Walk through a single log file and collect per-epoch metrics."""
    rows: List[Dict[str, float]] = []
    current_epoch: Optional[int] = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            epoch_match = EPOCH_RE.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            metrics = parse_metrics_line(line)
            if metrics is None or current_epoch is None:
                continue

            metrics["epoch"] = float(current_epoch)
            rows.append(metrics)

    return rows


def write_csv(rows: Iterable[Dict[str, float]], output_path: Path) -> None:
    """Serialise the collected metrics to `output_path`."""
    rows = list(rows)
    if not rows:
        return

    fieldnames = [
        "epoch",
        "steps_completed",
        "steps_total",
        "duration_s",
        "ms_per_step",
        "loss",
        "psnr",
        "val_loss",
        "val_psnr",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def process_logs(logs_root: Path, output_root: Path) -> List[Tuple[str, Path]]:
    """Parse every run directory beneath `logs_root` and emit CSVs."""
    emitted: List[Tuple[str, Path]] = []
    for run_dir in sorted(p for p in logs_root.iterdir() if p.is_dir()):
        log_files = sorted(run_dir.glob("run-simple-*.log"))
        if not log_files:
            continue
        log_path = log_files[-1]
        rows = extract_epoch_rows(log_path)
        if not rows:
            continue

        csv_path = output_root / run_dir.name / "epoch_metrics.csv"
        write_csv(rows, csv_path)
        emitted.append((run_dir.name, csv_path))
    return emitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Keras training logs into CSV tables.")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("Super_resolution/logs/experiment_1"),
        help="Directory holding the per-run folders with run-simple-*.log files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Super_resolution/experiments/experiment_1_constant_depth_3/csv_logs"),
        help="Where to write the per-run CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not logs_root.is_dir():
        raise SystemExit(f"Logs root not found: {logs_root}")

    emitted = process_logs(logs_root, output_root)
    if not emitted:
        print("No run-simple logs were converted.")
        return

    print("Generated CSV files:")
    for run_name, csv_path in emitted:
        print(f"  {run_name} -> {csv_path}")


if __name__ == "__main__":
    main()
