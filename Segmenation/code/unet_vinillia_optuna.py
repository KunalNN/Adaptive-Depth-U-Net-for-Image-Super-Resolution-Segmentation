#!/usr/bin/env python
"""
Optuna-based hyperparameter tuner for the ISIC-2017 U-Net training pipeline.

The script reuses the data loading and model utilities from ``unet_vinillia.py``
so that each trial trains the exact same architecture while exploring learning
rate, depth, base channels, batch size, and optional augmentation settings.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Make repo root importable
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset_paths import LOG_ROOT, MODEL_ROOT
from unet_vinillia import (
    DEFAULT_IMAGE_SUFFIX,
    DEFAULT_MASK_SUFFIX,
    DEFAULT_TRAIN_IMAGE_DIR,
    DEFAULT_TRAIN_MASK_DIR,
    DEFAULT_VAL_IMAGE_DIR,
    DEFAULT_VAL_MASK_DIR,
    build_dataset,
    build_unet,
    dice_coefficient,
    _discover_pairs,
)


@dataclass
class TrialConfig:
    learning_rate: float
    base_channels: int
    depth: int
    batch_size: int
    augment: bool
    seed: int
    run_name: str


class BestMetricCallback(tf.keras.callbacks.Callback):
    """Track the best value of a monitored metric during training."""

    def __init__(self, metric_name: str) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.best = float("-inf")

    def on_epoch_end(self, epoch: int, logs=None) -> None:  # type: ignore[override]
        logs = logs or {}
        value = logs.get(self.metric_name)
        if value is None:
            return
        value = float(value)
        if value > self.best:
            self.best = value


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_pairs(
    train_image_dir: Path,
    train_mask_dir: Path,
    val_image_dir: Path,
    val_mask_dir: Path,
    image_suffix: str,
    mask_suffix: str,
    limit_train: int | None,
    limit_val: int | None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_pairs = _discover_pairs(train_image_dir, train_mask_dir, image_suffix, mask_suffix, limit_train)
    val_pairs = _discover_pairs(val_image_dir, val_mask_dir, image_suffix, mask_suffix, limit_val)
    return train_pairs, val_pairs


def build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=args.pruner_startup_trials)
    if args.pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unsupported pruner: {args.pruner}")


def train_single_trial(
    cfg: TrialConfig,
    train_pairs: Sequence[Tuple[str, str]],
    val_pairs: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
    trial: optuna.trial.Trial | None,
    save_model: bool = False,
    artifact_dir: Path | None = None,
) -> float:
    tf.keras.backend.clear_session()
    set_global_seed(cfg.seed)

    train_ds = build_dataset(
        train_pairs,
        args.image_size,
        cfg.batch_size,
        shuffle=True,
        augment=cfg.augment,
        seed=cfg.seed,
    )
    val_ds = build_dataset(
        val_pairs,
        args.image_size,
        cfg.batch_size,
        shuffle=False,
        augment=False,
        seed=cfg.seed,
    )

    model = build_unet(
        input_size=args.image_size,
        num_classes=1,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
    )
    optimizer = Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            dice_coefficient,
        ],
    )

    callbacks: List[tf.keras.callbacks.Callback] = [
        EarlyStopping(
            monitor="val_dice_coefficient",
            patience=args.patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, args.patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        BestMetricCallback("val_dice_coefficient"),
    ]

    if trial is not None and args.pruner != "none":
        callbacks.append(TFKerasPruningCallback(trial, "val_dice_coefficient"))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.fit_verbose,
    )

    metric_tracker = next(cb for cb in callbacks if isinstance(cb, BestMetricCallback))
    best_metric = float(metric_tracker.best)

    if save_model and artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_path = artifact_dir / f"{cfg.run_name}_{timestamp}.keras"
        model.save(export_path)

    return best_metric


def objective_factory(
    args: argparse.Namespace,
    train_pairs: Sequence[Tuple[str, str]],
    val_pairs: Sequence[Tuple[str, str]],
):
    def objective(trial: optuna.trial.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        base_channels = trial.suggest_categorical("base_channels", args.base_channels_choices)
        depth = trial.suggest_int("depth", args.depth_min, args.depth_max)
        batch_size = trial.suggest_categorical("batch_size", args.batch_size_choices)
        augment = (
            trial.suggest_categorical("augment", [True, False])
            if args.tune_augment
            else args.augment
        )
        cfg = TrialConfig(
            learning_rate=learning_rate,
            base_channels=base_channels,
            depth=depth,
            batch_size=batch_size,
            augment=augment,
            seed=args.seed + trial.number,
            run_name=f"optuna_trial_{trial.number}",
        )
        return train_single_trial(cfg, train_pairs, val_pairs, args, trial)

    return objective


def save_results(path: Path, study: optuna.Study) -> None:
    best = study.best_trial
    payload = {
        "best_value": best.value,
        "best_params": best.params,
        "trial_number": best.number,
        "datetime": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for the U-Net segmentation model")
    parser.add_argument("--train_image_dir", type=Path, default=DEFAULT_TRAIN_IMAGE_DIR)
    parser.add_argument("--train_mask_dir", type=Path, default=DEFAULT_TRAIN_MASK_DIR)
    parser.add_argument("--val_image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR)
    parser.add_argument("--val_mask_dir", type=Path, default=DEFAULT_VAL_MASK_DIR)
    parser.add_argument("--image_suffix", type=str, default=DEFAULT_IMAGE_SUFFIX)
    parser.add_argument("--mask_suffix", type=str, default=DEFAULT_MASK_SUFFIX)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--augment", action="store_true", help="Use augmentations for every trial")
    parser.add_argument("--tune_augment", action="store_true", help="Let Optuna decide whether to apply augmentations")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///study.db")
    parser.add_argument("--load_if_exists", action="store_true")
    parser.add_argument("--pruner", choices=["none", "median", "hyperband"], default="median")
    parser.add_argument("--pruner_startup_trials", type=int, default=2)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=5e-4)
    parser.add_argument("--base_channels_choices", type=str, default="32,48,64")
    parser.add_argument("--batch_size_choices", type=str, default="4,8,12")
    parser.add_argument("--depth_min", type=int, default=3)
    parser.add_argument("--depth_max", type=int, default=5)
    parser.add_argument("--results_path", type=Path, default=LOG_ROOT / "optuna" / "unet_tuning.json")
    parser.add_argument("--save_best_model", action="store_true")
    parser.add_argument("--best_model_dir", type=Path, default=MODEL_ROOT / "optuna_best")
    parser.add_argument("--fit_verbose", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_channels_choices = parse_int_list(args.base_channels_choices)
    args.batch_size_choices = parse_int_list(args.batch_size_choices)
    if not args.base_channels_choices:
        raise ValueError("base_channels_choices may not be empty")
    if not args.batch_size_choices:
        raise ValueError("batch_size_choices may not be empty")
    if args.depth_min > args.depth_max:
        raise ValueError("depth_min must be <= depth_max")

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    train_image_dir = args.train_image_dir.expanduser()
    train_mask_dir = args.train_mask_dir.expanduser()
    val_image_dir = args.val_image_dir.expanduser()
    val_mask_dir = args.val_mask_dir.expanduser()

    for directory, label in [
        (train_image_dir, "training images"),
        (train_mask_dir, "training masks"),
        (val_image_dir, "validation images"),
        (val_mask_dir, "validation masks"),
    ]:
        if not directory.exists():
            raise FileNotFoundError(f"Missing {label} directory: {directory}")

    train_pairs, val_pairs = prepare_pairs(
        train_image_dir,
        train_mask_dir,
        val_image_dir,
        val_mask_dir,
        args.image_suffix,
        args.mask_suffix,
        args.limit_train,
        args.limit_val,
    )

    print(f"Prepared {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")

    pruner = build_pruner(args)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    objective = objective_factory(args, train_pairs, val_pairs)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best trial")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    if args.results_path:
        save_results(args.results_path, study)
        print(f"Results saved to {args.results_path}")

    if args.save_best_model:
        best_params = study.best_params
        cfg = TrialConfig(
            learning_rate=best_params["learning_rate"],
            base_channels=int(best_params["base_channels"]),
            depth=int(best_params["depth"]),
            batch_size=int(best_params["batch_size"]),
            augment=best_params.get("augment", args.augment),
            seed=args.seed,
            run_name="optuna_best",
        )
        metric = train_single_trial(
            cfg,
            train_pairs,
            val_pairs,
            args,
            trial=None,
            save_model=True,
            artifact_dir=args.best_model_dir,
        )
        print(f"Best model retrained and saved (val dice={metric:.4f}).")


if __name__ == "__main__":
    main()
