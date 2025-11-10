"""
Adaptive-depth U-Net training script for ISIC-2017 lesion segmentation.

The defaults follow the shared dataset path declarations so that a typical
cluster deployment pointing at /scratch/.../Segmenation/ISIC-2017 works
out-of-the-box, while still allowing overrides through CLI arguments.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.append(str(Path(__file__).resolve().parents[2])) # because Shared is two levels up

from shared.pipeline import sorted_alphanumeric
from dataset_paths import (
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VALID_IMAGE_DIR,
    VALID_MASK_DIR,
    MODEL_ROOT,
)

DEFAULT_TRAIN_IMAGE_DIR = TRAIN_IMAGE_DIR
DEFAULT_TRAIN_MASK_DIR = TRAIN_MASK_DIR
DEFAULT_VAL_IMAGE_DIR = VALID_IMAGE_DIR
DEFAULT_VAL_MASK_DIR = VALID_MASK_DIR
DEFAULT_MODEL_DIR = MODEL_ROOT
DEFAULT_IMAGE_SUFFIX = ".jpg"
DEFAULT_MASK_SUFFIX = "_segmentation.png"

AUTOTUNE = tf.data.AUTOTUNE


def conv_block(inputs: tf.Tensor, nf: int) -> tf.Tensor:
    # Make nf a trainable parameter for the Unet (maybe in future)
    x = L.Conv2D(nf, 3, padding="same", use_bias=True)(inputs) # Conv2D with 3x3 kernel with nf number of filters
    x = L.LayerNormalization(axis=-1)(x) # LayerNorm over channels
    x = L.Activation("relu")(x) # ReLU activation

    # Running this twice deepens the receptive field and injects nonlinearity while keeping spatial size unchanged
    x = L.Conv2D(nf, 3, padding="same", use_bias=True)(x) 
    x = L.LayerNormalization(axis=-1)(x) 
    x = L.Activation("relu")(x)
    return x


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #


def encoder_block(x: tf.Tensor, nf: int) -> Tuple[tf.Tensor, tf.Tensor]:
    skip = conv_block(x, nf)
    pooled = L.MaxPooling2D(2)(skip)
    return pooled, skip


def decoder_block(x: tf.Tensor, skip: tf.Tensor, nf: int) -> tf.Tensor:
    x = L.Conv2DTranspose(nf, 2, strides=2, padding="same")(x)
    x = L.Concatenate()([x, skip])
    return conv_block(x, nf)


def build_unet(input_size: int, num_classes: int = 1, base_channels: int = 32, depth: int = 4) -> Model:
    inputs = Input(shape=(input_size, input_size, 3), name="images")
    skips: List[tf.Tensor] = []
    x = inputs
    nf = base_channels

    for _ in range(depth):
        x, skip = encoder_block(x, nf)
        skips.append(skip)
        nf *= 2

    x = conv_block(x, nf)

    for skip in reversed(skips):
        nf //= 2
        x = decoder_block(x, skip, nf)

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = L.Conv2D(num_classes, 1, activation=activation, name="mask_logits")(x)
    return Model(inputs, outputs, name="unet_isic_baseline")


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred) + smooth
    denominator = tf.reduce_sum(y_true + y_pred) + smooth
    return numerator / denominator


def _canonical_key(path: Path) -> str:
    stem = path.stem.lower()
    replacements = [
        "_segmentation",
        "_mask",
        "_leftimg8bit",
        "_gtfine_labelids",
        "_gtfine_polygons",
        "_gtfine_color",
        "_gtfine_instanceids",
        "_gtcoarse_labelids",
        "_gtcoarse_color",
        "_gtcoarse_instanceids",
        "_instanceids",
    ]
    for token in replacements:
        stem = stem.replace(token, "")
    return stem


def _discover_pairs(
    image_dir: Path,
    mask_dir: Path,
    image_suffix: str,
    mask_suffix: str,
    limit: int | None,
) -> List[Tuple[str, str]]:
    image_candidates = [
        str(p) for p in image_dir.rglob(f"*{image_suffix}")
        if p.is_file()
    ]
    image_paths = [Path(p) for p in sorted_alphanumeric(image_candidates)]

    mask_lookup = {
        _canonical_key(p): p
        for p in mask_dir.rglob(f"*{mask_suffix}")
        if p.is_file()
    }

    if not image_paths:
        raise ValueError(f"No images found in {image_dir} with suffix {image_suffix}")
    if not mask_lookup:
        raise ValueError(f"No masks found in {mask_dir} with suffix {mask_suffix}")

    pairs: List[Tuple[str, str]] = []
    for image_path in image_paths:
        key = _canonical_key(image_path)
        mask_path = mask_lookup.get(key)
        if mask_path is None:
            raise ValueError(f"Missing mask for image {image_path.name} (expected key {key})")
        pairs.append((str(image_path), str(mask_path)))

    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def _parse_example(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (image_size, image_size), method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((image_size, image_size, 3))

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.image.resize(mask, (image_size, image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask


def build_dataset(
    pairs: Sequence[Tuple[str, str]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda img, msk: _parse_example(img, msk, image_size), num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline U-Net on the ISIC-2017 dataset.")
    parser.add_argument("--train_image_dir", type=Path, default=DEFAULT_TRAIN_IMAGE_DIR, help="Directory of training images.")
    parser.add_argument("--train_mask_dir", type=Path, default=DEFAULT_TRAIN_MASK_DIR, help="Directory of training segmentation masks.")
    parser.add_argument("--val_image_dir", type=Path, default=DEFAULT_VAL_IMAGE_DIR, help="Directory of validation images.")
    parser.add_argument("--val_mask_dir", type=Path, default=DEFAULT_VAL_MASK_DIR, help="Directory of validation masks.")
    parser.add_argument("--image_suffix", type=str, default=DEFAULT_IMAGE_SUFFIX, help="Suffix/pattern for image files.")
    parser.add_argument("--mask_suffix", type=str, default=DEFAULT_MASK_SUFFIX, help="Suffix/pattern for mask files.")
    parser.add_argument("--image_size", type=int, default=256, help="Square input resolution.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--base_channels", type=int, default=32, help="Number of filters in the first encoder block.")
    parser.add_argument("--depth", type=int, default=4, help="Depth of the encoder/decoder.")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory to save checkpoints.")
    parser.add_argument("--run_name", type=str, default="unet_isic", help="Prefix for saved checkpoints.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for shuffling.")
    parser.add_argument("--limit_train", type=int, default=None, help="Optional limit on number of training samples.")
    parser.add_argument("--limit_val", type=int, default=None, help="Optional limit on number of validation samples.")
    parser.add_argument("--augment", action="store_true", help="Enable simple geometric augmentations.")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision policy if GPUs support it.")
    parser.add_argument("--fit_verbose", type=int, choices=[0, 1, 2], default=2, help="Keras verbosity mode (0=silent, 1=progress bar, 2=per-epoch).")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
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

    train_pairs = _discover_pairs(train_image_dir, train_mask_dir, args.image_suffix, args.mask_suffix, args.limit_train)
    val_pairs = _discover_pairs(val_image_dir, val_mask_dir, args.image_suffix, args.mask_suffix, args.limit_val)

    train_ds = build_dataset(train_pairs, args.image_size, args.batch_size, shuffle=True, augment=args.augment, seed=args.seed)
    val_ds = build_dataset(val_pairs, args.image_size, args.batch_size, shuffle=False, augment=False, seed=args.seed)

    print(f"Loaded {len(train_pairs)} training samples and {len(val_pairs)} validation samples.")

    model = build_unet(args.image_size, num_classes=1, base_channels=args.base_channels, depth=args.depth)
    optimizer = Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        dice_coefficient,
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.model_dir / f"{args.run_name}_best.keras"
    print(f"Checkpoints will be written to {checkpoint_path}")

    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_dice_coefficient", mode="max", save_best_only=True, save_weights_only=False, verbose=1),
        EarlyStopping(monitor="val_dice_coefficient", patience=10, mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.fit_verbose,
    )

    final_path = args.model_dir / f"{args.run_name}_final.keras"
    model.save(final_path)


if __name__ == "__main__":
    train(parse_args())
