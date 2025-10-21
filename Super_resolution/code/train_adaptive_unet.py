"""
Adaptive-depth U-Net training script.

This script consolidates the adaptive-depth notebook workflow into a reusable
entry point that can be launched on a cluster. It parameterises the scale
factor, produces low-resolution inputs on-the-fly, and keeps the encoder depth
in line with the design table from the project summary.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2])) # because Shared is two levels up

import argparse
import glob
import re
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, mixed_precision
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import BackupAndRestore, EarlyStopping, ModelCheckpoint

from shared.custom_layers import ClipAdd, ResizeByScale, ResizeToMatch, estimate_bottleneck_size, infer_depth_from_scale
from shared.pipeline import degrade_image, load_rgb_image, make_tf_dataset

# Science cluster enables XLA globally; Resize ops lack an XLA kernel, so disable JIT.
tf.config.optimizer.set_jit(False)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Updated defaults to match the scratch copy of DIV2K prepared for this project.
DATA_ROOT = Path("/scratch/knarwani/Final_data/Super_resolution")
DEFAULT_HIGH_RES_DIR = DATA_ROOT / "DIV2K_train_HR"
DEFAULT_LOW_RES_DIR = DATA_ROOT / "DIV2K_train_LR_bicubic-2" / "X4"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "Super_resolution" / "models"


# --------------------------------------------------------------------------- #
# Local data utilities (decoupled from shared/pipeline)
# --------------------------------------------------------------------------- #

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


def load_and_preprocess_image_tf(path: tf.Tensor, size: int) -> tf.Tensor:
    """Read, decode, normalise, and resize an image using TensorFlow ops."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.AREA)
    image.set_shape((size, size, 3))
    return image


def degrade_to_lr_tf(hr_image: tf.Tensor, scale: float, output_size: int) -> tf.Tensor:
    """Create a synthetic low-resolution counterpart from an HR tensor."""
    scale = tf.convert_to_tensor(scale, dtype=tf.float32)
    base = tf.cast(output_size, tf.float32)
    down_size = tf.cast(tf.round(scale * base), tf.int32)
    down_size = tf.maximum(down_size, 1)
    resized = tf.image.resize(hr_image, [down_size, down_size], method=tf.image.ResizeMethod.AREA)
    restored = tf.image.resize(resized, [output_size, output_size], method=tf.image.ResizeMethod.BICUBIC)
    return tf.clip_by_value(restored, 0.0, 1.0)


def build_dataset(
    pairs: Sequence[Tuple[str, str]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    scale: float,
    hr_size: int,
) -> tf.data.Dataset | None:
    """Construct a tf.data pipeline that yields (lr, hr) batches."""
    if not pairs:
        return None

    hr_files = [hr for hr, _ in pairs]
    lr_files = [lr for _, lr in pairs]

    ds = tf.data.Dataset.from_tensor_slices((hr_files, lr_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(hr_files), seed=seed, reshuffle_each_iteration=True)

    def load_pair(hr_path: tf.Tensor, lr_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        hr_image = load_and_preprocess_image_tf(hr_path, hr_size)

        def load_lr_from_disk() -> tf.Tensor:
            return load_and_preprocess_image_tf(lr_path, hr_size)

        def synthesize_lr() -> tf.Tensor:
            return degrade_to_lr_tf(hr_image, scale, hr_size)

        lr_image = tf.cond(
            tf.strings.length(lr_path) > 0,
            true_fn=load_lr_from_disk,
            false_fn=synthesize_lr,
        )
        return lr_image, hr_image

    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def canonical_hr_key(path: str | Path) -> str:
    """Return a normalised key for HR filenames (e.g. 0001.png -> 0001)."""
    return Path(path).stem.lower()


def canonical_lr_key(path: str | Path) -> str:
    """
    Return a normalised key for LR filenames.

    DIV2K LR files include the scale suffix (e.g. 0001x4.png). Strip the trailing
    `xN` token so that we can align them to the HR counterparts.
    """
    stem = Path(path).stem.lower()
    stem = re.sub(r"x\d+$", "", stem)
    return stem


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

def build_super_resolution_unet(
    scale: float,
    base_channels: int = 64,
    residual_head_channels: int = 64,
    depth_override: int | None = None,
    input_size: int = 256,
) -> Tuple[Model, Dict[str, object]]:
    
    # Pick encoder depth explicitly or infer from the downscale factor
    depth = depth_override or infer_depth_from_scale(scale)

    down_layer = ResizeByScale(scale, name="enc_down") # shrinks features in the encoder
    up_layer = ResizeToMatch(name="dec_up") # upsamples to a skip-connection’s size in the decoder 

    # Low-resolution RGB enters here, decoder will reuse stored skips.
    inputs = Input(shape=(input_size, input_size, 3), name="low_res_input") 
    # when calling Input(), it creates a placeholder tensor that represents the input data for the model

    skips: List[tf.Tensor] = []
    x = inputs
    nf = base_channels

    # Encoder: extract features, remember skips, and shrink spatially.
    for _ in range(depth):
        skip = conv_block(x, nf)
        pooled = down_layer(skip)
        skips.append(skip)
        x = pooled
        nf *= 2

    # Bottleneck operates at smallest spatial resolution with widest channels.
    x = conv_block(x, nf)

    # Decoder: upsample, fuse skip features, and refine activations.
    for skip in reversed(skips):
        nf //= 2
        x = up_layer([x, skip]) # resize to match skip's spatial dimensions
        x = L.Conv2D(nf, 3, padding="same", activation="relu")(x) # conv to reduce artifacts after upsampling
        # upsampling by resize tends to introduce checkerboard or ringing artifacts. Running a quick Conv2D + ReLU immediately after the resize cleans up those artifacts and pre‑conditions the feature map before you fuse it with the skip tensor
        x = L.Concatenate()([x, skip]) # fuse skip connection
        x = conv_block(x, nf)

    # Residual head predicts RGB correction; zero init keeps identity start.
    x_head = conv_block(x, residual_head_channels) # to shrink features width
    # maps those features to three channels (RGB) with a 1×1 convolution. Zero kernel/bias init means the network starts by outputting zeros, so initially the model just copies the input until training teaches it useful corrections.
    residual = L.Conv2D(
        3,
        1,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="residual_rgb",
    )(x_head)
    # ClipAdd merges residual with the input while clamping to image range.
    outputs = ClipAdd(name="enhanced_rgb")([inputs, residual])

    # Return the assembled model plus diagnostics for logging/reporting.
    model = Model(inputs, outputs, name=f"U-Net_SR_scale{scale:.2f}_depth{depth}")
    info = {
        "scale": scale,
        "depth": depth,
        "bottleneck_size": estimate_bottleneck_size(input_size, scale, depth),
        "base_channels": base_channels,
    }
    return model, info


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #

def build_losses_and_metrics() -> Tuple[tf.keras.losses.Loss, List[tf.keras.metrics.Metric]]:
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
        feat_true = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_true * 255.0))
        feat_pred = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0))
        return tf.reduce_mean(tf.square(feat_true - feat_pred))

    def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse_val = tf.cast(mse_loss(y_true, y_pred), tf.float32)
        ssim_val = tf.cast(ssim_loss(y_true, y_pred), tf.float32)
        perc_val = tf.cast(perceptual_loss(y_true, y_pred), tf.float32)
        total = alpha * mse_val + beta * ssim_val + gamma * perc_val
        return total

    def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    combined_loss.__name__ = "combined_loss"
    psnr_metric.__name__ = "psnr"
    return combined_loss, [psnr_metric]


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    high_res_dir = Path(args.high_res_dir).expanduser()
    if not high_res_dir.exists():
        raise FileNotFoundError(f"High-resolution directory not found: {high_res_dir}")

    hr_paths = sorted_alphanumeric(
        glob.glob(str(high_res_dir / f"*{args.image_suffix}"))
    )
    if args.limit and args.limit > 0:
        hr_paths = hr_paths[:args.limit]
    if not hr_paths:
        raise ValueError("No high-resolution images found with the given suffix.")

    hr_images = np.stack([load_rgb_image(path, args.hr_size) for path in hr_paths])

    if args.low_res_dir:
        low_res_dir = Path(args.low_res_dir).expanduser()
        if not low_res_dir.exists():
            # Allow pointing at the parent directory that contains scale-specific folders.
            candidates = [p for p in low_res_dir.glob("X*") if p.is_dir()]
            if len(candidates) == 1:
                low_res_dir = candidates[0]
            else:
                raise FileNotFoundError(f"Low-resolution directory not found: {low_res_dir}")

        low_paths = sorted_alphanumeric(glob.glob(str(low_res_dir / f"*{args.image_suffix}")))
        low_index = {canonical_lr_key(path): path for path in low_paths}

        lr_images = []
        for hr_path in hr_paths:
            key = canonical_hr_key(hr_path)
            low_path = low_index.get(key)
            if low_path is None:
                raise ValueError(f"Missing low-resolution counterpart for {hr_path}")
            lr_images.append(load_rgb_image(low_path, args.hr_size))
        lr_images = np.stack(lr_images)
    else:
        lr_images = np.stack([degrade_image(img, args.scale, args.hr_size) for img in hr_images])

    train_split = 1.0 - (args.val_split + args.test_split)
    if train_split <= 0:
        raise ValueError("Validation and test splits leave no room for training data.")

    train_idx, val_idx, test_idx = split_indices(
        len(hr_images), train_split, args.val_split, args.test_split, args.seed
    )

    train_ds = make_tf_dataset(lr_images, hr_images, train_idx, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_tf_dataset(lr_images, hr_images, val_idx, args.batch_size, shuffle=False, seed=args.seed)
    test_ds = make_tf_dataset(lr_images, hr_images, test_idx, args.batch_size, shuffle=False, seed=args.seed) if len(test_idx) else None

    if args.mixed_precision:
        available_gpus = tf.config.list_physical_devices("GPU")
        if not available_gpus:
            print("[warn] Mixed precision requested but no GPU detected; running in float32.")
            args.mixed_precision = False
        else:
            mixed_precision.set_global_policy("mixed_float16")

    model, info = build_super_resolution_unet(
        scale=args.scale,
        base_channels=args.base_channels,
        residual_head_channels=args.residual_head_channels,
        depth_override=args.depth_override,
        input_size=args.hr_size,
    )

    loss_fn, metrics = build_losses_and_metrics()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn,
        metrics=metrics,
        jit_compile=False,  # avoid XLA forcing unsupported resize ops on the cluster
    )

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"unet_adaptive_scale{args.scale:.2f}_depth{info['depth']}.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
        BackupAndRestore(str(model_dir / "train_backup")),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    print("Training complete.")
    print(f"Model info: {info}")
    print(f"Checkpoint saved to: {ckpt_path}")

    eval_targets = [("Validation", val_ds)]
    if test_ds is not None:
        eval_targets.append(("Test", test_ds))

    for name, dataset in eval_targets:
        psnr_vals, ssim_vals, msssim_vals = [], [], []
        n_images = 0
        for lr_batch, hr_batch in dataset:
            pred = model(lr_batch, training=False)
            pred = tf.cast(tf.clip_by_value(pred, 0.0, 1.0), tf.float32)
            hr = tf.cast(hr_batch, tf.float32)

            psnr_vals.append(tf.image.psnr(hr, pred, max_val=1.0).numpy())
            ssim_vals.append(tf.image.ssim(hr, pred, max_val=1.0).numpy())
            msssim_vals.append(tf.image.ssim_multiscale(hr, pred, max_val=1.0).numpy())
            n_images += int(hr_batch.shape[0])

        def mean_std(values: List[np.ndarray]) -> Tuple[float, float]:
            arr = np.concatenate(values, axis=0).astype(np.float64)
            return float(np.mean(arr)), float(np.std(arr))

        if not psnr_vals:
            print(f"{name}: no samples, skipping metric aggregation.")
            continue

        m_psnr, s_psnr = mean_std(psnr_vals)
        m_ssim, s_ssim = mean_std(ssim_vals)
        m_msssim, s_msssim = mean_std(msssim_vals)

        print(f"{name} samples evaluated: {n_images}")
        print(f"  PSNR    : {m_psnr:.4f} ± {s_psnr:.4f} dB")
        print(f"  SSIM    : {m_ssim:.4f} ± {s_ssim:.4f}")
        print(f"  MS-SSIM : {m_msssim:.4f} ± {s_msssim:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive-depth U-Net for super-resolution.")
    parser.add_argument("--scale", type=float, required=True, help="Downscale factor (0 < scale < 1).")
    parser.add_argument("--high_res_dir", type=str, default=str(DEFAULT_HIGH_RES_DIR), help="Directory with HR images.")
    parser.add_argument("--low_res_dir", type=str, default=str(DEFAULT_LOW_RES_DIR), help="Optional directory with LR images; leave blank to enable synthetic degradation.")
    parser.add_argument("--image_suffix", type=str, default=".png", help="Image suffix to filter HR files.")
    parser.add_argument("--hr_size", type=int, default=256, help="Square crop size for HR images.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit the number of training samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dataset shuffling and splitting.")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--residual_head_channels", type=int, default=64)
    parser.add_argument("--depth_override", type=int, default=None, help="Force a specific encoder depth.")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed_float16 policy.")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Directory to store checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
