from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

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


def load_rgb_image_full(path: str | Path) -> np.ndarray:
    """Read an image from disk, convert to RGB, and normalise to [0, 1] without resizing."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def random_patch(
    image: np.ndarray,
    patch_size: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Extract a single random patch of shape (patch_size, patch_size, 3)."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an HxWx3 RGB array.")

    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError("patch_size exceeds image dimensions.")

    generator = rng or np.random.default_rng()
    max_y = height - patch_size
    max_x = width - patch_size
    top = int(generator.integers(0, max_y + 1)) if max_y > 0 else 0
    left = int(generator.integers(0, max_x + 1)) if max_x > 0 else 0
    return image[top : top + patch_size, left : left + patch_size, :]


def random_patches(
    image: np.ndarray,
    patch_size: int,
    count: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Extract multiple random patches stacked along axis 0."""
    if count <= 0:
        raise ValueError("count must be positive.")
    generator = rng or np.random.default_rng()
    patches = [
        random_patch(image, patch_size, rng=generator)
        for _ in range(count)
    ]
    return np.stack(patches, axis=0)


def grid_patches(
    image: np.ndarray,
    patch_size: int,
    *,
    stride: int | None = None,
    drop_remainder: bool = False,
) -> np.ndarray:
    """Extract a regular grid of patches across the image."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an HxWx3 RGB array.")

    stride = stride or patch_size
    if stride <= 0:
        raise ValueError("stride must be positive.")

    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError("patch_size exceeds image dimensions.")

    patches: List[np.ndarray] = []
    max_y = height - patch_size
    max_x = width - patch_size
    y_range = range(0, max_y + 1, stride)
    x_range = range(0, max_x + 1, stride)

    for top in y_range:
        for left in x_range:
            patches.append(image[top : top + patch_size, left : left + patch_size, :])

    if not patches and not drop_remainder:
        # If stride skipped the entire image, include the bottom-right aligned patch.
        patches.append(image[-patch_size:, -patch_size:, :])

    return np.stack(patches, axis=0) if patches else np.empty((0, patch_size, patch_size, 3), dtype=image.dtype)


def _iter_random_patch_pairs(
    hr_files: Sequence[str],
    patch_size: int,
    patches_per_image: int,
    scale: float,
    seed: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    hr_files = list(hr_files)
    if not hr_files:
        return
    while True:
        rng.shuffle(hr_files)
        for path in hr_files:
            hr_image = load_rgb_image_full(path)
            hr_patches = random_patches(hr_image, patch_size, count=patches_per_image, rng=rng)
            for hr_patch in hr_patches:
                lr_patch = degrade_image(hr_patch, scale, patch_size)
                yield lr_patch, hr_patch


def _iter_grid_patch_pairs(
    hr_files: Sequence[str],
    patch_size: int,
    stride: int,
    scale: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for path in hr_files:
        hr_image = load_rgb_image_full(path)
        hr_patches = grid_patches(hr_image, patch_size, stride=stride, drop_remainder=False)
        if hr_patches.size == 0:
            continue
        for hr_patch in hr_patches:
            lr_patch = degrade_image(hr_patch, scale, patch_size)
            yield lr_patch, hr_patch


def make_training_patch_dataset(
    hr_files: Sequence[str],
    patch_size: int,
    patches_per_image: int,
    scale: float,
    batch_size: int,
    seed: int,
    shuffle_buffer: int = 1024,
) -> Tuple[tf.data.Dataset, int]:
    """
    Create an infinite tf.data pipeline that yields random (lr, hr) patch batches.
    """
    hr_files = list(hr_files)
    if not hr_files:
        raise ValueError("hr_files must contain at least one path.")
    if patches_per_image <= 0:
        raise ValueError("patches_per_image must be positive.")

    total_patches = len(hr_files) * patches_per_image

    def generator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield from _iter_random_patch_pairs(hr_files, patch_size, patches_per_image, scale, seed)

    output_signature = (
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, total_patches


def make_eval_patch_dataset(
    hr_files: Sequence[str],
    patch_size: int,
    scale: float,
    batch_size: int,
    *,
    stride: int | None = None,
) -> Tuple[tf.data.Dataset, int, List[str]]:
    """
    Create a finite tf.data pipeline of grid patches for evaluation.
    Returns the dataset, total patch count, and an ordered list of patch identifiers.
    """
    hr_files = list(hr_files)
    if not hr_files:
        raise ValueError("hr_files must contain at least one path.")
    stride = stride or patch_size
    if stride <= 0:
        raise ValueError("stride must be positive.")

    def generator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield from _iter_grid_patch_pairs(hr_files, patch_size, stride, scale)

    output_signature = (
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    patch_labels: List[str] = []
    for path in hr_files:
        image = load_rgb_image_full(path)
        patches = grid_patches(image, patch_size, stride=stride, drop_remainder=False)
        stem = Path(path).name
        for idx in range(patches.shape[0]):
            patch_labels.append(f"{stem}#patch{idx:04d}")

    total_patches = len(patch_labels)
    return dataset, total_patches, patch_labels


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
