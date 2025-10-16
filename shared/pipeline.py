from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf


def sorted_alphanumeric(items: Iterable[str]) -> List[str]:
    """Sort strings so that entries with trailing numbers follow numeric order."""

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


def load_rgb_image(path: str | Path, size: int) -> np.ndarray:
    """Read an image from disk, convert to RGB, resize, and normalise to [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def degrade_image(image: np.ndarray, scale: float, output_size: int) -> np.ndarray:
    """Synthesise a low-resolution image by shrinking and re-upscaling an HR sample."""
    if not 0 < scale < 1:
        raise ValueError("Scale must be between 0 and 1 for degradation.")

    hr = np.asarray(image, dtype=np.float32)
    hr = np.clip(hr, 0.0, 1.0)
    height, width = hr.shape[:2]
    target_h = target_w = output_size if output_size > 0 else max(height, width)

    down_h = max(1, int(round(target_h * scale)))
    down_w = max(1, int(round(target_w * scale)))

    downsampled = cv2.resize(hr, (down_w, down_h), interpolation=cv2.INTER_AREA)
    upsampled = cv2.resize(downsampled, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return upsampled.astype(np.float32)


def split_indices(n_samples: int, train: float, val: float, test: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test using provided fractions."""
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
    ds = tf.data.Dataset.from_tensor_slices((lr_images[indices], hr_images[indices]))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
