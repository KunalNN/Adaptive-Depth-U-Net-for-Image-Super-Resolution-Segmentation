#!/usr/bin/env python3
"""
Evaluate a saved Adaptive-Depth U-Net checkpoint on the ISIC-2017 test split.

This script reuses the dataset utilities and model builder from train_adaptive_unet.py
so that checkpoints trained via run_experiment_* scripts can be measured on the held-out
test set after training finishes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow importing dataset_paths

from dataset_paths import (  # noqa: E402
    LOG_ROOT,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
)
from train_adaptive_unet import (  # noqa: E402
    PROTOCOLS,
    DEFAULT_BASE_CHANNELS,
    DEFAULT_DEPTH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    build_adaptive_depth_unet,
    build_isic_dataset,
    dice_metric,
    iou_metric,
    set_global_seed,
)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a JSON object")
    return data


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_int(name: str, cli_value: int, config: Dict[str, Any], default: int) -> int:
    if cli_value and cli_value > 0:
        return cli_value
    from_config = coerce_int(config.get(name))
    if from_config is not None and from_config > 0:
        return from_config
    return default


def resolve_path(cli_value: Optional[str], config_value: Optional[str], fallback: Path) -> Path:
    if cli_value:
        return Path(cli_value).expanduser()
    if config_value:
        return Path(config_value).expanduser()
    return Path(fallback).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Adaptive-Depth U-Net checkpoint on the ISIC test split.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved checkpoint (e.g., .weights.h5 produced by training).",
    )
    parser.add_argument("--config", help="Optional config.json produced during training to reuse hyperparameters.")
    parser.add_argument("--run_name", help="Override run name (defaults to config run_name or checkpoint stem).")
    parser.add_argument("--output_dir", help="Directory for evaluation artifacts (defaults to logs/evaluations/<run>).")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS.keys()), help="Protocol key to select the evaluation loss.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for evaluation (defaults to config/protocol).")
    parser.add_argument("--image_size", type=int, default=0, help="Input resolution (defaults to config or training default).")
    parser.add_argument("--base_channels", type=int, default=0, help="Base channel width (defaults to config or training default).")
    parser.add_argument("--depth", type=int, default=0, help="Encoder/decoder depth (defaults to config or training default).")
    parser.add_argument("--test_images", help="Override test image directory.")
    parser.add_argument("--test_masks", help="Override test mask directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for deterministic dataset ordering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_protocol = (config.get("protocol") or "").upper() if config.get("protocol") else None
    protocol_key = (args.protocol or config_protocol or "A").upper()
    if protocol_key not in PROTOCOLS:
        raise ValueError(f"Unknown protocol '{protocol_key}'. Expected one of: {', '.join(sorted(PROTOCOLS.keys()))}")

    run_name = args.run_name or config.get("run_name") or checkpoint_path.stem

    image_size = resolve_int("image_size", args.image_size, config, DEFAULT_IMAGE_SIZE)
    base_channels = resolve_int("base_channels", args.base_channels, config, DEFAULT_BASE_CHANNELS)
    depth = resolve_int("depth", args.depth, config, DEFAULT_DEPTH)

    default_batch_size = PROTOCOLS[protocol_key].batch_size
    batch_size = resolve_int("batch_size", args.batch_size, config, default_batch_size)

    test_images = resolve_path(args.test_images, config.get("test_images"), TEST_IMAGE_DIR)
    test_masks = resolve_path(args.test_masks, config.get("test_masks"), TEST_MASK_DIR)

    if not test_images.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_images}")
    if not test_masks.exists():
        raise FileNotFoundError(f"Test mask directory not found: {test_masks}")

    output_root = Path(args.output_dir).expanduser() if args.output_dir else Path(LOG_ROOT) / "evaluations" / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    test_ds, test_count = build_isic_dataset(
        image_dir=test_images,
        mask_dir=test_masks,
        batch_size=batch_size,
        image_size=image_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )
    print(f"[eval] Test split contains {test_count} samples")

    model = build_adaptive_depth_unet(
        input_size=image_size,
        base_channels=base_channels,
        depth=depth,
    )
    loss_fn = PROTOCOLS[protocol_key].loss_builder()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=[dice_metric, iou_metric])

    try:
        model.load_weights(str(checkpoint_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load weights from {checkpoint_path}: {exc}") from exc

    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    results = {
        "evaluated_at": timestamp,
        "checkpoint": str(checkpoint_path),
        "run_name": run_name,
        "protocol": protocol_key,
        "image_size": image_size,
        "base_channels": base_channels,
        "depth": depth,
        "batch_size": batch_size,
        "test_images": str(test_images),
        "test_masks": str(test_masks),
        "test_samples": test_count,
        "metrics": metrics,
    }

    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    summary_path = output_root / "metrics.txt"
    with summary_path.open("w") as handle:
        handle.write(f"Evaluation run: {run_name}\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Evaluated at: {timestamp}\n")
        handle.write(f"Samples: {test_count}\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {value:.6f}\n")

    print("[eval] Metrics written to", metrics_path)
    for key, value in metrics.items():
        print(f"[eval] {key}: {value:.4f}")


if __name__ == "__main__":
    main()
