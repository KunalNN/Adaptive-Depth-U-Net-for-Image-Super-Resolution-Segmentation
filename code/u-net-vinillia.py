#!/usr/bin/env python3
"""
Legacy U-Net baseline for single-image super-resolution.

This refactors the original notebook-exported code into a small CLI so the
baseline can be reproduced without manual cell execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers as L
from tensorflow.keras.callbacks import BackupAndRestore, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def sorted_alphanumeric(items: Iterable[str]) -> List[str]:
    """Sort strings so that 10 follows 9 instead of 1."""

    def tokenize(token: str):
        return int(token) if token.isdigit() else token.lower()

    def split_key(text: str):
        token = ""
        tokens: List[str] = []
        for char in text:
            if char.isdigit():
                if token and not token[-1].isdigit():
                    tokens.append(token)
                    token = ""
                token += char
            else:
                if token and token[-1].isdigit():
                    tokens.append(token)
                    token = ""
                token += char
        if token:
            tokens.append(token)
        return [tokenize(part) for part in tokens]

    return sorted(items, key=split_key)


def load_image_stack(directory: Path, size: int, limit: int | None = None) -> np.ndarray:
    """Load and normalise images from a directory into an array of shape (N, H, W, 3)."""
    paths = sorted_alphanumeric([p.name for p in directory.iterdir() if p.is_file()])
    if limit is not None:
        paths = paths[:limit]

    images: List[np.ndarray] = []
    for filename in paths:
        img = cv2.imread(str(directory / filename), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {directory / filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    if not images:
        raise ValueError(f"No images found in {directory}")

    return np.stack(images, axis=0)


def split_indices(n_samples: int, train: float, val: float, test: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test using the provided fractions."""
    if not 0 < train < 1:
        raise ValueError("Train fraction should be between 0 and 1.")
    if not 0 <= val < 1 or not 0 <= test < 1:
        raise ValueError("Val/test fractions should be between 0 and 1.")
    total = train + val + test
    if total <= 0:
        raise ValueError("Fractions must sum to a positive value.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_count = int(round(n_samples * train / total))
    val_count = int(round(n_samples * val / total))
    train_count = min(train_count, n_samples - 2) if n_samples > 2 else train_count
    val_count = min(val_count, n_samples - train_count - 1) if n_samples > (train_count + 1) else val_count
    test_count = n_samples - train_count - val_count

    if train_count <= 0:
        raise ValueError("Train split is empty; adjust fractions.")

    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]
    return train_idx, val_idx, test_idx


def make_tf_dataset(
    lr_images: np.ndarray,
    hr_images: np.ndarray,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a tf.data pipeline for the given subset indices."""
    ds = tf.data.Dataset.from_tensor_slices((lr_images[indices], hr_images[indices]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

def conv_block(inputs: tf.Tensor, nf: int) -> tf.Tensor:
    x = L.Conv2D(nf, 3, padding="same")(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(nf, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x


def encoder_block(inputs: tf.Tensor, nf: int) -> Tuple[tf.Tensor, tf.Tensor]:
    x = conv_block(inputs, nf)
    pooled = L.MaxPool2D(pool_size=(2, 2))(x)
    return x, pooled


def decoder_block(inputs: tf.Tensor, skip: tf.Tensor, nf: int) -> tf.Tensor:
    x = L.UpSampling2D(size=(2, 2), interpolation="bilinear")(inputs)
    x = L.Conv2D(nf, 3, padding="same", activation="relu")(x)
    x = L.Concatenate()([x, skip])
    return conv_block(x, nf)


def build_super_resolution_unet(input_shape: Tuple[int, int, int]) -> Model:
    inputs = Input(shape=input_shape, name="low_res_input")

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    bottleneck = conv_block(p4, 1024)

    d1 = decoder_block(bottleneck, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = L.Conv2D(3, 1, padding="same", activation="sigmoid", name="enhanced_rgb")(d4)
    return Model(inputs, outputs, name="U-Net_SR_256x256")


# -----------------------------------------------------------------------------
# Losses & metrics
# -----------------------------------------------------------------------------

def build_losses() -> Tuple[tf.keras.losses.Loss, List[tf.keras.metrics.Metric]]:
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    vgg.trainable = False
    feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv4").output)

    alpha = tf.constant(1.0, dtype=tf.float32)
    beta = tf.constant(0.1, dtype=tf.float32)
    gamma = tf.constant(0.01, dtype=tf.float32)

    def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    def perceptual_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
        ft = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_true * 255.0))
        fp = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0))
        return tf.reduce_mean(tf.square(ft - fp))

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse_val = tf.cast(mse_loss(y_true, y_pred), tf.float32)
        ssim_val = tf.cast(ssim_loss(y_true, y_pred), tf.float32)
        perc_val = tf.cast(perceptual_loss(y_true, y_pred), tf.float32)
        return alpha * mse_val + beta * ssim_val + gamma * perc_val

    def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    combined_loss.__name__ = "combined_loss"
    psnr_metric.__name__ = "psnr"
    return combined_loss, [psnr_metric]


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def evaluate(model: Model, dataset: tf.data.Dataset) -> Dict[str, Tuple[float, float]]:
    psnr_vals, ssim_vals, msssim_vals = [], [], []
    for lr_batch, hr_batch in dataset:
        preds = tf.cast(tf.clip_by_value(model(lr_batch, training=False), 0.0, 1.0), tf.float32)
        hr_batch = tf.cast(hr_batch, tf.float32)
        psnr_vals.append(tf.image.psnr(hr_batch, preds, max_val=1.0).numpy())
        ssim_vals.append(tf.image.ssim(hr_batch, preds, max_val=1.0).numpy())
        msssim_vals.append(tf.image.ssim_multiscale(hr_batch, preds, max_val=1.0).numpy())

    if not psnr_vals:
        return {}

    def mean_std(values: List[np.ndarray]) -> Tuple[float, float]:
        arr = np.concatenate(values, axis=0).astype(np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    return {
        "psnr": mean_std(psnr_vals),
        "ssim": mean_std(ssim_vals),
        "ms_ssim": mean_std(msssim_vals),
    }


def main(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)

    hr_images = load_image_stack(Path(args.high_res_dir), args.hr_size, limit=args.limit)
    lr_images = load_image_stack(Path(args.low_res_dir), args.hr_size, limit=args.limit)
    if hr_images.shape != lr_images.shape:
        raise ValueError("High-resolution and low-resolution stacks must align one-to-one.")

    train_idx, val_idx, test_idx = split_indices(
        n_samples=hr_images.shape[0],
        train=args.train_split,
        val=args.val_split,
        test=args.test_split,
        seed=args.seed,
    )

    train_ds = make_tf_dataset(lr_images, hr_images, train_idx, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_tf_dataset(lr_images, hr_images, val_idx, args.batch_size, shuffle=False, seed=args.seed)
    test_ds = make_tf_dataset(lr_images, hr_images, test_idx, args.batch_size, shuffle=False, seed=args.seed) if len(test_idx) else None

    model = build_super_resolution_unet((args.hr_size, args.hr_size, 3))
    loss_fn, metrics = build_losses()
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=loss_fn, metrics=metrics)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(Path(args.model_dir) / "unet_vanilla_best.keras"), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        BackupAndRestore(str(Path(args.model_dir) / "train_backup")),
    ]

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    print("Validation metrics:", evaluate(model, val_ds))
    if test_ds is not None:
        print("Test metrics:", evaluate(model, test_ds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vanilla super-resolution U-Net baseline.")
    parser.add_argument("--high_res_dir", type=str, required=True, help="Directory containing high-resolution images.")
    parser.add_argument("--low_res_dir", type=str, required=True, help="Directory containing low-resolution images.")
    parser.add_argument("--hr_size", type=int, default=256, help="Input/output spatial size.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.8, help="Relative portion of samples for training.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Relative portion of samples for validation.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Relative portion of samples for testing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs to load.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to store checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
