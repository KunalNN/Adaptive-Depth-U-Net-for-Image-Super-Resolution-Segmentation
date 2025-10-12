"""
Adaptive-depth U-Net training script.

This script consolidates the adaptive-depth notebook workflow into a reusable
entry point that can be launched on a cluster. It parameterises the scale
factor, produces low-resolution inputs on-the-fly, and keeps the encoder depth
in line with the design table from the project summary.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import Input, Model, mixed_precision
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import BackupAndRestore, EarlyStopping, ModelCheckpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIGH_RES_DIR = PROJECT_ROOT / "dataset" / "Raw Data" / "high_res"
DEFAULT_LOW_RES_DIR = PROJECT_ROOT / "dataset" / "Raw Data" / "low_res"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def sorted_alphanumeric(paths: Iterable[str]) -> List[str]:
    """Sort file paths in human order so that 10 follows 9 and not 1."""

    def convert(token: str):
        return int(token) if token.isdigit() else token.lower()

    def key_func(path: str):
        parts: List[str] = []
        token = ""
        for char in path:
            if char.isdigit():
                if token and not token[-1].isdigit():
                    parts.append(token)
                    token = ""
                token += char
            else:
                if token and token[-1].isdigit():
                    parts.append(token)
                    token = ""
                token += char
        if token:
            parts.append(token)
        return [convert(p) for p in parts]

    return sorted(paths, key=key_func)


def load_rgb_image(path: str, size: int) -> np.ndarray:
    """Load an RGB image, resize to `size`, and normalise to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def degrade_image(hr_image: np.ndarray, scale: float, size: int) -> np.ndarray:
    """Downscale by `scale` then bicubic upsample back to `size`."""
    low_size = max(1, int(round(size * scale)))
    low = cv2.resize(hr_image, (low_size, low_size), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    return restored


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test using the provided fractions."""
    if n == 0:
        raise ValueError("No samples available after loading dataset.")
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_val = min(n_val, n - 1) if n > 1 else 0
    n_test = min(n_test, n - n_val - 1) if n > (n_val + 1) else 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions leave no samples for training.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_tf_dataset(
    lr_images: np.ndarray,
    hr_images: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Create a tf.data pipeline for a given index set."""
    data = tf.data.Dataset.from_tensor_slices((lr_images[indices], hr_images[indices]))
    if shuffle:
        data = data.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    return data.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Depth heuristic
# --------------------------------------------------------------------------- #

def infer_depth_from_scale(scale: float, min_depth: int = 1, max_depth: int = 4) -> int:
    """
    Decide encoder depth using the project design table:
      scale <= 0.25 -> depth 1
      scale <= 0.45 -> depth 2
      otherwise     -> depth 3
    """
    if not (0.05 < scale < 1.0):
        raise ValueError("Scale should be between 0 and 1 (exclusive).")

    if scale <= 0.25:
        depth = 1
    elif scale <= 0.45:
        depth = 2
    else:
        depth = 3

    depth = max(min_depth, min(depth, max_depth))
    return depth


def estimate_bottleneck_size(hr: int, scale: float, depth: int) -> int:
    """Compute the spatial extent at the bottleneck for diagnostics."""
    size = hr
    for _ in range(depth):
        size = max(1, int(round(size * scale)))
    return size


# --------------------------------------------------------------------------- #
# Custom layers
# --------------------------------------------------------------------------- #

@register_keras_serializable(package="resize")
class ResizeByScale(L.Layer):
    def __init__(self, scale: float, method: str = "bilinear", antialias: bool = True, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.method = method
        self.antialias = antialias

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_dtype = x.dtype # save the incoming tensor's dtype
        h = tf.shape(x)[1] # height
        w = tf.shape(x)[2] # width
        nh = tf.cast(tf.round(tf.cast(h, tf.float32) * self.scale), tf.int32) # new height
        nw = tf.cast(tf.round(tf.cast(w, tf.float32) * self.scale), tf.int32) # new width

        # Resize the image
        res = tf.image.resize(tf.cast(x, tf.float32), [nh, nw], method=self.method, antialias=self.antialias)
        return tf.cast(res, x_dtype) # cast back to original dtype

    # this method allow Keres to later recreate the layer with the same settings 
    def get_config(self) -> Dict[str, object]:
        return {
            **super().get_config(),
            "scale": self.scale,
            "method": self.method,
            "antialias": self.antialias,
        }


@register_keras_serializable(package="resize")
class ResizeToMatch(L.Layer):
    def __init__(self, method: str = "bilinear", antialias: bool = True, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.antialias = antialias

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, ref = inputs # x is the tensor to resize, ref is the reference tensor
        target_hw = tf.shape(ref)[1:3] # target height and width
        res = tf.image.resize(tf.cast(x, tf.float32), target_hw, method=self.method, antialias=self.antialias)
        return tf.cast(res, x.dtype)

    def get_config(self) -> Dict[str, object]:
        return {
            **super().get_config(),
            "method": self.method,
            "antialias": self.antialias,
        }


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


@register_keras_serializable(package="utils")
class ClipAdd(L.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        inp, residual = inputs # inp is the low-res input, residual is the predicted residual
        out = tf.cast(inp, tf.float32) + tf.cast(residual, tf.float32) # add in float32 for numerical stability
        #  The network takes the low-res image as input, learns to predict the residual correction—the difference between the degraded image and the desired high-res output—and Clipadd adds that correction back to the original input 
        return tf.cast(tf.clip_by_value(out, 0.0, 1.0), inp.dtype) 


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
            raise FileNotFoundError(f"Low-resolution directory not found: {low_res_dir}")

        low_paths = sorted_alphanumeric(glob.glob(str(low_res_dir / f"*{args.image_suffix}")))
        low_index = {Path(path).name: path for path in low_paths}

        lr_images = []
        for hr_path in hr_paths:
            key = Path(hr_path).name
            low_path = low_index.get(key)
            if low_path is None:
                raise ValueError(f"Missing low-resolution counterpart for {hr_path}")
            lr_images.append(load_rgb_image(low_path, args.hr_size))
        lr_images = np.stack(lr_images)
    else:
        lr_images = np.stack([degrade_image(img, args.scale, args.hr_size) for img in hr_images])

    train_idx, val_idx, test_idx = split_indices(
        len(hr_images), args.val_split, args.test_split, args.seed
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
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--residual_head_channels", type=int, default=64)
    parser.add_argument("--depth_override", type=int, default=None, help="Force a specific encoder depth.")
    # parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed_float16 policy.")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Directory to store checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
