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
import json
import math
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, mixed_precision
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import BackupAndRestore, EarlyStopping, ModelCheckpoint

from dataset_paths import HR_TRAIN_DIR, LR_TRAIN_DIR, LOG_ROOT, MODEL_ROOT
from shared.custom_layers import ClipAdd, ResizeByScale, ResizeToMatch, estimate_bottleneck_size, custom_depth_from_scale
from shared.pipeline import (
    load_rgb_image_full,
    make_eval_patch_dataset,
    make_training_patch_dataset,
    random_patches,
    degrade_image,
)

# Science cluster enables XLA globally; Resize ops lack an XLA kernel, so disable JIT.
tf.config.optimizer.set_jit(False)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Updated defaults to match the scratch copy of DIV2K prepared for this project.
DEFAULT_HIGH_RES_DIR = HR_TRAIN_DIR
DEFAULT_LOW_RES_DIR = LR_TRAIN_DIR
DEFAULT_MODEL_DIR = MODEL_ROOT
DEFAULT_LOG_DIR = LOG_ROOT
DEFAULT_IMAGE_SUFFIX = ".png"
DEFAULT_HR_SIZE = 256
DEFAULT_BASE_CHANNELS = 64
DEFAULT_RESIDUAL_HEAD_CHANNELS = 64

# --- constants ---
DATA_LR_SHRINK = 0.5  # To ensure that the low-resolution (LR) input remains consistent across all scales.


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


def rgb_to_luma_bt601(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an RGB tensor in [0, 1] to its BT.601 luminance channel in [0, 1].

    Parameters
    ----------
    image
        Tensor shaped (N, H, W, 3) or (H, W, 3) containing RGB values normalised to [0, 1].
    """
    image = tf.cast(image, tf.float32)
    coeffs = tf.constant([65.481, 128.553, 24.966], dtype=tf.float32)
    coeffs = tf.reshape(coeffs, [1, 1, 1, 3])
    y_channel = tf.reduce_sum(image * coeffs, axis=-1, keepdims=True) + 16.0
    return tf.clip_by_value(y_channel / 255.0, 0.0, 1.0)


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
    max_depth: int = 7,
) -> Tuple[Model, Dict[str, object]]:
    
    # Pick encoder depth explicitly or infer from the downscale factor
    depth = (
        depth_override
        if depth_override is not None
        else custom_depth_from_scale(scale, max_depth=max_depth, base_resolution=input_size)
    )

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
        "max_depth": max_depth,
    }
    return model, info


# --------------------------------------------------------------------------- #
# Losses & metrics
# --------------------------------------------------------------------------- #

def build_losses_and_metrics(loss_name: str) -> Tuple[tf.keras.losses.Loss, List[tf.keras.metrics.Metric]]:
    """
    Create the supervised loss and metrics used during training.

    Parameters
    ----------
    loss_name
        Identifier for the training loss. Supported values:
        - "charbonnier": robust L1 variant commonly used for PSNR benchmarks.
        - "l1": mean absolute error on [0, 1] RGB.
        - "combined": original perceptual + SSIM + MSE cocktail.
    """
    loss_key = loss_name.lower()

    def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    if loss_key == "charbonnier":
        epsilon = tf.constant(1e-3, dtype=tf.float32)

        def charbonnier_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            diff = y_true - y_pred
            return tf.reduce_mean(tf.sqrt(tf.square(diff) + tf.square(epsilon)))

        charbonnier_loss.__name__ = "charbonnier_loss"
        psnr_metric.__name__ = "psnr"
        return charbonnier_loss, [psnr_metric]

    if loss_key == "l1":
        def l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            return tf.reduce_mean(tf.abs(y_true - y_pred))

        l1_loss.__name__ = "l1_loss"
        psnr_metric.__name__ = "psnr"
        return l1_loss, [psnr_metric]

    if loss_key == "combined":
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

        combined_loss.__name__ = "combined_loss"
        psnr_metric.__name__ = "psnr"
        return combined_loss, [psnr_metric]

    raise ValueError(f"Unknown loss '{loss_name}'. Expected one of: 'charbonnier', 'l1', 'combined'.")


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    image_suffix = DEFAULT_IMAGE_SUFFIX
    patch_size = args.patch_size
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
    if args.patches_per_image <= 0:
        raise ValueError("patches_per_image must be positive.")
    if args.eval_stride is not None and args.eval_stride <= 0:
        raise ValueError("eval_stride must be positive when provided.")
    if args.shuffle_buffer < 0:
        raise ValueError("shuffle_buffer must be non-negative.")
    if args.preview_patches < 0:
        raise ValueError("preview_patches must be non-negative.")

    hr_size = patch_size
    base_channels = DEFAULT_BASE_CHANNELS
    residual_head_channels = DEFAULT_RESIDUAL_HEAD_CHANNELS

    if args.max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    if args.initial_epoch < 0:
        raise ValueError("initial_epoch must be non-negative.")
    if args.initial_epoch >= args.epochs:
        raise ValueError("initial_epoch must be smaller than --epochs to resume training.")

    high_res_dir_input = args.high_res_dir or DEFAULT_HIGH_RES_DIR
    high_res_dir = Path(high_res_dir_input).expanduser()
    if not high_res_dir.exists():
        raise FileNotFoundError(f"High-resolution directory not found: {high_res_dir}")

    hr_paths = sorted_alphanumeric(
        glob.glob(str(high_res_dir / f"*{image_suffix}"))
    )
    if args.limit and args.limit > 0:
        hr_paths = hr_paths[:args.limit]
    if not hr_paths:
        raise ValueError("No high-resolution images found with the given suffix.")

    if args.low_res_dir:
        print("[info] --low_res_dir is ignored in patch mode; LR patches are generated on the fly.")

    train_split = 1.0 - (args.val_split + args.test_split)
    if train_split <= 0:
        raise ValueError("Validation and test splits leave no room for training data.")

    train_idx, val_idx, test_idx = split_indices(
        len(hr_paths), train_split, args.val_split, args.test_split, args.seed
    )

    train_paths = [hr_paths[i] for i in train_idx]
    val_paths = [hr_paths[i] for i in val_idx]
    test_paths = [hr_paths[i] for i in test_idx]

    train_ds, train_patch_count = make_training_patch_dataset(
        train_paths,
        patch_size=patch_size,
        patches_per_image=args.patches_per_image,
        scale= DATA_LR_SHRINK,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )

    val_fit_ds = None
    val_patch_count = 0
    if val_paths:
        val_fit_ds, val_patch_count, _ = make_eval_patch_dataset(
            val_paths,
            patch_size=patch_size,
            scale=DATA_LR_SHRINK,
            batch_size=args.batch_size,
            stride=args.eval_stride,
        )
        val_fit_ds = val_fit_ds.repeat()

    test_patch_count = 0
    if test_paths:
        _, test_patch_count, _ = make_eval_patch_dataset(
            test_paths,
            patch_size=patch_size,
            scale=DATA_LR_SHRINK,
            batch_size=args.batch_size,
            stride=args.eval_stride,
        )

    steps_per_epoch = math.ceil(train_patch_count / args.batch_size)
    if steps_per_epoch <= 0:
        raise ValueError("Training dataset produced zero patches. Check patches_per_image or dataset splits.")
    val_steps = math.ceil(val_patch_count / args.batch_size) if val_patch_count else None

    if args.mixed_precision:
        available_gpus = tf.config.list_physical_devices("GPU")
        if not available_gpus:
            print("[warn] Mixed precision requested but no GPU detected; running in float32.")
            args.mixed_precision = False
        else:
            mixed_precision.set_global_policy("mixed_float16")

    model, info = build_super_resolution_unet(
        scale=args.scale,
        base_channels=base_channels,
        residual_head_channels=residual_head_channels,
        depth_override=args.depth_override,
        input_size=hr_size,
        max_depth=args.max_depth,
    )

    loss_fn, metrics = build_losses_and_metrics(args.loss)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn,
        metrics=metrics,
        jit_compile=False,  # avoid XLA forcing unsupported resize ops on the cluster
    )

    resume_checkpoint: Path | None = None
    if args.resume_from:
        candidate = Path(args.resume_from).expanduser()
        if candidate.is_dir():
            ckpts = sorted(candidate.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                raise FileNotFoundError(
                    f"--resume_from directory {candidate} contains no '.keras' checkpoints."
                )
            resume_checkpoint = ckpts[0]
        else:
            if not candidate.exists():
                raise FileNotFoundError(f"Checkpoint not found: {candidate}")
            resume_checkpoint = candidate

    if resume_checkpoint is not None:
        print(f"[info] Loading weights from {resume_checkpoint}")
        try:
            model.load_weights(str(resume_checkpoint))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load checkpoint from {resume_checkpoint}") from exc
        if args.initial_epoch == 0:
            print("[warn] --resume_from supplied without --initial_epoch; training will restart from epoch 0.")
    elif args.initial_epoch > 0:
        print(
            "[warn] --initial_epoch was set without --resume_from; training will skip the initial epochs but start from random weights."
        )

    summary_lines: List[str] = []
    model.summary(print_fn=summary_lines.append)
    for line in summary_lines:
        print(line)

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / f"unet_adaptive_scale_new_loss{args.scale:.2f}_depth{info['depth']}.keras"

    log_root = Path(args.log_dir).expanduser()
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inferred_name = f"scale{args.scale:.2f}_bs{args.batch_size}_lr{args.learning_rate:.0e}_{timestamp}"
    run_name = args.run_name or inferred_name
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "scale": args.scale,
        "depth": info["depth"],
        "max_depth": args.max_depth,
        "patch_size": patch_size,
        "patches_per_image": args.patches_per_image,
        "eval_stride": args.eval_stride or patch_size,
        "base_channels": base_channels,
        "residual_head_channels": residual_head_channels,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "train_images": int(len(train_paths)),
        "val_images": int(len(val_paths)),
        "test_images": int(len(test_paths)),
        "train_patches_per_epoch": int(train_patch_count),
        "val_patches": int(val_patch_count),
        "test_patches": int(test_patch_count),
        "steps_per_epoch": int(steps_per_epoch),
        "validation_steps": int(val_steps) if val_steps is not None else None,
        "mixed_precision": bool(args.mixed_precision),
        "high_res_dir": str(high_res_dir),
        "low_res_mode": "synthetic_patches",
        "model_dir": str(model_dir),
        "log_dir": str(run_dir),
        "created_at": timestamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    (run_dir / "model_summary.txt").write_text("\n".join(summary_lines))

    preview_count = min(args.preview_patches, len(train_paths))
    writer = tf.summary.create_file_writer(str(run_dir))
    with writer.as_default():
        tf.summary.text("config/hyperparameters", tf.constant(json.dumps(config_payload, indent=2)), step=0)
        tf.summary.scalar("dataset/images/train", len(train_paths), step=0)
        tf.summary.scalar("dataset/images/val", len(val_paths), step=0)
        tf.summary.scalar("dataset/images/test", len(test_paths), step=0)
        tf.summary.scalar("dataset/patches_per_epoch/train", train_patch_count, step=0)
        tf.summary.scalar("dataset/patches/val", val_patch_count, step=0) 
        tf.summary.scalar("dataset/patches/test", test_patch_count, step=0)
        if preview_count > 0 and train_paths:
            rng = np.random.default_rng(args.seed)
            preview_hr_path = train_paths[0]
            preview_hr_image = load_rgb_image_full(preview_hr_path)
            hr_preview_np = random_patches(preview_hr_image, patch_size, count=preview_count, rng=rng)
            lr_preview_np = np.stack(
                [degrade_image(patch, DATA_LR_SHRINK, patch_size) for patch in hr_preview_np],
                axis=0,
            )
            hr_preview = tf.convert_to_tensor(hr_preview_np)
            lr_preview = tf.convert_to_tensor(lr_preview_np)
            tf.summary.image("samples/hr_train", hr_preview, step=0, max_outputs=preview_count)
            tf.summary.image("samples/lr_train", lr_preview, step=0, max_outputs=preview_count)
            tf.summary.histogram("hist/hr_train", tf.reshape(hr_preview, [-1]), step=0)
            tf.summary.histogram("hist/lr_train", tf.reshape(lr_preview, [-1]), step=0)
        tf.summary.text("model/summary", tf.constant("\n".join(summary_lines)), step=0)
    writer.flush()

    custom_dir = Path(run_dir) / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(custom_dir))  # << not run_dir

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(run_dir),      # Keras will write to run_dir/train and run_dir/validation
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
        update_freq='epoch',       # make it explicit
    )

    backup_dir = run_dir / "train_backup"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
        BackupAndRestore(str(backup_dir)),
        tensorboard_callback,
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_fit_ds if val_fit_ds is not None else None,
        validation_steps=val_steps if val_fit_ds is not None else None,
        validation_freq=1,
        callbacks=callbacks,
        verbose=2,
    )

    print("Training complete.")
    print(f"Model info: {info}")
    print(f"Checkpoint saved to: {ckpt_path}")

    eval_targets: List[Tuple[str, tf.data.Dataset]] = []
    if val_paths:
        val_eval_ds, _, _ = make_eval_patch_dataset(
            val_paths,
            patch_size=patch_size,
            scale= DATA_LR_SHRINK,
            batch_size=args.batch_size,
            stride=args.eval_stride,
        )
        eval_targets.append(("Validation", val_eval_ds))
    if test_paths:
        test_eval_ds, _, _ = make_eval_patch_dataset(
            test_paths,
            patch_size=patch_size,
            scale=DATA_LR_SHRINK,
            batch_size=args.batch_size,
            stride=args.eval_stride,
        )
        eval_targets.append(("Test", test_eval_ds))

    if args.eval_shave is not None:
        eval_shave = max(0, int(args.eval_shave))
    else:
        inv_scale = 1.0 / args.scale if args.scale > 0 else 0.0
        scale_factor = int(round(inv_scale)) if inv_scale > 0 else 0
        eval_shave = 2 * scale_factor if scale_factor > 0 else 0

    if eval_shave * 2 >= hr_size and hr_size > 0:
        adjusted = max(0, (hr_size // 2) - 1)
        print(
            f"[warn] eval_shave={eval_shave} removes the full frame for hr_size={hr_size}; "
            f"reducing to {adjusted} pixels."
        )
        eval_shave = adjusted

    for name, dataset in eval_targets:
        psnr_vals, ssim_vals, msssim_vals, mse_vals = [], [], [], []
        n_patches = 0
        for lr_batch, hr_batch in dataset:
            pred_rgb = model(lr_batch, training=False)
            pred_rgb = tf.cast(tf.clip_by_value(pred_rgb, 0.0, 1.0), tf.float32)
            hr_rgb = tf.cast(hr_batch, tf.float32)

            pred_y = rgb_to_luma_bt601(pred_rgb)
            hr_y = rgb_to_luma_bt601(hr_rgb)

            if eval_shave > 0:
                pred_y = pred_y[:, eval_shave:-eval_shave, eval_shave:-eval_shave, :]
                hr_y = hr_y[:, eval_shave:-eval_shave, eval_shave:-eval_shave, :]

            psnr_vals.append(tf.image.psnr(hr_y, pred_y, max_val=1.0).numpy())
            ssim_vals.append(tf.image.ssim(hr_y, pred_y, max_val=1.0).numpy())
            msssim_vals.append(tf.image.ssim_multiscale(hr_y, pred_y, max_val=1.0).numpy())
            mse_vals.append(
                tf.reduce_mean(tf.square(hr_y - pred_y), axis=[1, 2, 3]).numpy()
            )
            n_patches += int(hr_batch.shape[0])

        def mean_std(values: List[np.ndarray]) -> Tuple[float, float]:
            arr = np.concatenate(values, axis=0).astype(np.float64)
            return float(np.mean(arr)), float(np.std(arr))

        if not psnr_vals:
            print(f"{name}: no samples, skipping metric aggregation.")
            continue

        m_psnr, s_psnr = mean_std(psnr_vals)
        m_ssim, s_ssim = mean_std(ssim_vals)
        m_msssim, s_msssim = mean_std(msssim_vals)
        m_mse, s_mse = mean_std(mse_vals)

        print(f"{name} patches evaluated: {n_patches}")
        print(f"  MSE(Y)     : {m_mse:.6f} ± {s_mse:.6f}")
        print(f"  PSNR(Y)    : {m_psnr:.4f} ± {s_psnr:.4f} dB")
        print(f"  SSIM(Y)    : {m_ssim:.4f} ± {s_ssim:.4f}")
        print(f"  MS-SSIM(Y) : {m_msssim:.4f} ± {s_msssim:.4f}")
        eval_step = history.epoch[-1] if hasattr(history, "epoch") and history.epoch else args.epochs
        with writer.as_default():
            tag_prefix = name.lower()
            tf.summary.scalar(f"eval/{tag_prefix}_mse_y", m_mse, step=eval_step)
            tf.summary.scalar(f"eval/{tag_prefix}_psnr_y", m_psnr, step=eval_step)
            tf.summary.scalar(f"eval/{tag_prefix}_ssim_y", m_ssim, step=eval_step)
            tf.summary.scalar(f"eval/{tag_prefix}_msssim_y", m_msssim, step=eval_step)
    writer.flush()
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive-depth U-Net for super-resolution.")
    parser.add_argument("--scale", type=float, required=True, help="Downscale factor (0 < scale < 1).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        type=str,
        default="charbonnier",
        choices=["charbonnier", "l1", "combined"],
        help="Training loss to optimise. Use 'charbonnier' or 'l1' for PSNR-focused benchmarks.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit the number of training samples.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dataset shuffling and splitting.")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_HR_SIZE, help="Side length for HR/LR training patches.")
    parser.add_argument(
        "--patches_per_image",
        type=int,
        default=4,
        help="Random patches sampled per image for each training epoch.",
    )
    parser.add_argument(
        "--eval_stride",
        type=int,
        default=None,
        help="Stride to use when tiling evaluation patches (defaults to patch_size).",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=1024,
        help="Shuffle buffer size for the training patch dataset.",
    )
    parser.add_argument(
        "--preview_patches",
        type=int,
        default=3,
        help="Number of training patches to log to TensorBoard at step 0.",
    )
    parser.add_argument(
        "--eval_shave",
        type=int,
        default=None,
        help="Pixels to trim from each border before computing PSNR/SSIM (default: 2 * round(1 / scale)).",
    )
    parser.add_argument("--depth_override", type=int, default=None, help="Force a specific encoder depth.")
    parser.add_argument(
        "--max_depth",
        type=int,
        default=7,
        help="Maximum encoder depth when inferring from scale (ignored if --depth_override is set).",
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed_float16 policy.")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Directory to store checkpoints.")
    parser.add_argument("--log_dir", type=str, default=str(DEFAULT_LOG_DIR), help="Directory to store TensorBoard logs.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional explicit run name for TensorBoard.")
    parser.add_argument("--high_res_dir", type=str, default=None, help="Override the high-resolution dataset directory.")
    parser.add_argument(
        "--low_res_dir",
        type=str,
        default=None,
        help="Ignored in patch mode; low-resolution patches are synthesised on the fly.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional path to a .keras checkpoint (or directory containing checkpoints) to resume from.",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=0,
        help="Epoch index to begin training from when resuming (must be < --epochs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
