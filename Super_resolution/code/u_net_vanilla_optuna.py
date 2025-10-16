#!/usr/bin/env python3
"""
Optuna-driven hyperparameter tuning for the vanilla super-resolution U-Net.

This script searches learning rate and loss weights, then trains a final model
with the best-found configuration.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2])) # because Shared is two levels up

from shared.pipeline import load_image_stack, split_indices, sorted_alphanumeric

import argparse
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tensorflow.keras import Input, Model, layers as L
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


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


def build_feature_extractor(input_size: int) -> Model:
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3))
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv4").output)


def make_combined_loss(alpha: float, beta: float, gamma: float, feature_extractor: Model):
    a = tf.constant(alpha, dtype=tf.float32)
    b = tf.constant(beta, dtype=tf.float32)
    g = tf.constant(gamma, dtype=tf.float32)

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
        ft = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_true * 255.0), training=False)
        fp = feature_extractor(tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0), training=False)
        return tf.reduce_mean(tf.square(ft - fp))

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse_val = tf.cast(mse_loss(y_true, y_pred), tf.float32)
        ssim_val = tf.cast(ssim_loss(y_true, y_pred), tf.float32)
        perc_val = tf.cast(perceptual_loss(y_true, y_pred), tf.float32)
        return a * mse_val + b * ssim_val + g * perc_val

    loss.__name__ = "combined_loss"
    return loss


def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


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


# -----------------------------------------------------------------------------
# Optuna optimisation
# -----------------------------------------------------------------------------

def run_study(
    args: argparse.Namespace,
    lr_images: np.ndarray,
    hr_images: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()

        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        alpha = trial.suggest_float("alpha", 0.5, 2.0)
        beta = trial.suggest_float("beta", 1e-3, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)

        train_ds = make_tf_dataset(lr_images, hr_images, train_idx, batch_size, shuffle=True, seed=args.seed)
        val_ds = make_tf_dataset(lr_images, hr_images, val_idx, batch_size, shuffle=False, seed=args.seed)

        feature_extractor = build_feature_extractor(args.hr_size)
        model = build_super_resolution_unet((args.hr_size, args.hr_size, 3))
        loss_fn = make_combined_loss(alpha, beta, gamma, feature_extractor)
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=[psnr_metric])

        pruning_cb = TFKerasPruningCallback(trial, monitor="val_loss")
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stop_patience, restore_best_weights=True, verbose=0)
        tmp_dir = tempfile.mkdtemp()
        checkpoint = ModelCheckpoint(
            filepath=str(Path(tmp_dir) / "best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.max_tuning_epochs,
            callbacks=[early_stop, checkpoint, pruning_cb],
            verbose=0,
        )

        return float(np.min(history.history["val_loss"]))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_warmup_steps=args.pruner_warmup),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=args.show_progress)
    return study


def train_final_model(
    args: argparse.Namespace,
    lr_images: np.ndarray,
    hr_images: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    best_params: Dict[str, float],
) -> Model:
    batch_size = int(best_params["batch_size"])
    lr = float(best_params["lr"])
    alpha = float(best_params["alpha"])
    beta = float(best_params["beta"])
    gamma = float(best_params["gamma"])

    train_ds = make_tf_dataset(lr_images, hr_images, train_idx, batch_size, shuffle=True, seed=args.seed)
    val_ds = make_tf_dataset(lr_images, hr_images, val_idx, batch_size, shuffle=False, seed=args.seed)

    feature_extractor = build_feature_extractor(args.hr_size)
    model = build_super_resolution_unet((args.hr_size, args.hr_size, 3))
    loss_fn = make_combined_loss(alpha, beta, gamma, feature_extractor)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn, metrics=[psnr_metric])

    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.final_patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(model_dir / "unet_vanilla_optuna_best.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        epochs=args.final_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )
    return model


def main(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)
    model_dir = Path(args.model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    hr_images = load_image_stack(Path(args.high_res_dir).expanduser(), args.hr_size, limit=args.limit)
    lr_images = load_image_stack(Path(args.low_res_dir).expanduser(), args.hr_size, limit=args.limit)
    if hr_images.shape != lr_images.shape:
        raise ValueError("High-resolution and low-resolution stacks must align one-to-one.")

    train_idx, val_idx, test_idx = split_indices(
        n_samples=hr_images.shape[0],
        train=args.train_split,
        val=args.val_split,
        test=args.test_split,
        seed=args.seed,
    )

    study = run_study(args, lr_images, hr_images, train_idx, val_idx)
    print("Best val_loss:", study.best_value)
    print("Best params:", study.best_params)

    final_model = train_final_model(args, lr_images, hr_images, train_idx, val_idx, study.best_params)

    if len(val_idx):
        val_ds = make_tf_dataset(lr_images, hr_images, val_idx, args.eval_batch, shuffle=False, seed=args.seed)
        print("Validation metrics:", evaluate(final_model, val_ds))

    if len(test_idx):
        test_ds = make_tf_dataset(lr_images, hr_images, test_idx, args.eval_batch, shuffle=False, seed=args.seed)
        print("Test metrics:", evaluate(final_model, test_ds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and train vanilla SR U-Net with Optuna.")
    parser.add_argument("--high_res_dir", type=str, required=True, help="Directory containing high-resolution images.")
    parser.add_argument("--low_res_dir", type=str, required=True, help="Directory containing low-resolution images.")
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4], help="Batch sizes to sample during tuning.")
    parser.add_argument("--max_tuning_epochs", type=int, default=40)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--pruner_warmup", type=int, default=5)
    parser.add_argument("--show_progress", action="store_true", help="Show Optuna study progress bar.")
    parser.add_argument("--final_epochs", type=int, default=100)
    parser.add_argument("--final_patience", type=int, default=15)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs to load.")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
