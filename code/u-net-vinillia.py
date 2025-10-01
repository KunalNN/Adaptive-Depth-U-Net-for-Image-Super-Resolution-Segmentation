#!/usr/bin/env python
# coding: utf-8
# Minimal U-Net training on fixed dataset paths (cluster-ready, no internet)

import os, re
from tqdm import tqdm
import numpy as np
np.random.seed(0)

import cv2
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, Activation,
                                     BatchNormalization, UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# 1) DATA LOADING (fixed)
# =========================
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r'([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

SIZE = 256

# High-res
high_img = []
hr_path = '/scratch/knarwani/super-resolution/dataset/Raw Data/high_res'
files = sorted_alphanumeric(os.listdir(hr_path))
for fname in tqdm(files, desc="HR"):
    if fname == '855.png': break
    img = cv2.imread(os.path.join(hr_path, fname), cv2.IMREAD_COLOR)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    high_img.append(img_to_array(img))

# Low-res
low_img = []
lr_path = '/scratch/knarwani/super-resolution/dataset/Raw Data/low_res'
files = sorted_alphanumeric(os.listdir(lr_path))
for fname in tqdm(files, desc="LR"):
    if fname == '855.png': break
    img = cv2.imread(os.path.join(lr_path, fname), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    low_img.append(img_to_array(img))

# Splits (unchanged)
train_high_image = np.asarray(high_img[:700], dtype=np.float32)
train_low_image  = np.asarray( low_img[:700], dtype=np.float32)

validation_high_image = np.asarray(high_img[700:810], dtype=np.float32)
validation_low_image  = np.asarray( low_img[700:810], dtype=np.float32)

test_high_image = np.asarray(high_img[810:], dtype=np.float32)
test_low_image  = np.asarray( low_img[810:], dtype=np.float32)

print("TF:", tf.__version__)
print("Shape of training images:", train_high_image.shape)
print("Shape of test images:",      test_high_image.shape)
print("Shape of validation images:",validation_high_image.shape)

# =========================
# 2) MODEL
# =========================
def conv_block(inputs, n):
    x = Conv2D(n, 3, padding="same")(inputs); x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Conv2D(n, 3, padding="same")(x);      x = BatchNormalization()(x); x = Activation("relu")(x)
    return x

def enc(inputs, n):
    x = conv_block(inputs, n); p = MaxPool2D((2,2))(x); return x, p

def dec(inputs, skip, n):
    x = UpSampling2D()(inputs)
    x = Conv2D(n, 3, padding="same", activation="relu")(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, n)
    return x

def build_unet(input_shape=(256,256,3)):
    inp = Input(shape=input_shape)
    s1,p1 = enc(inp,  64)
    s2,p2 = enc(p1,  128)
    s3,p3 = enc(p2,  256)
    s4,p4 = enc(p3,  512)
    b     = conv_block(p4, 1024)
    d1    = dec(b,  s4, 512)
    d2    = dec(d1, s3, 256)
    d3    = dec(d2, s2, 128)
    d4    = dec(d3, s1,  64)
    out   = Conv2D(3, 1, padding="same", activation="sigmoid")(d4)
    return Model(inp, out, name="U-Net_SR_x1")

# =========================
# 3) LOSSES / METRICS
# =========================
def mse_loss(y_true, y_pred):  return tf.reduce_mean(tf.square(y_true - y_pred))
def ssim_loss(y_true, y_pred): return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
def psnr_metric(y_true, y_pred): return tf.image.psnr(y_true, y_pred, max_val=1.0)
def ssim_metric(y_true, y_pred): return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
ALPHA, BETA = 1.0, 0.1
def combined_loss(y_true, y_pred): return ALPHA*mse_loss(y_true,y_pred) + BETA*ssim_loss(y_true,y_pred)

# =========================
# 4) DATASETS
# =========================
BATCH_SIZE = 4
EPOCHS = 100
train_ds = (tf.data.Dataset.from_tensor_slices((train_low_image, train_high_image))
            .shuffle(len(train_low_image)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
valid_ds = (tf.data.Dataset.from_tensor_slices((validation_low_image, validation_high_image))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# =========================
# 5) TRAIN
# =========================
model = build_unet((SIZE,SIZE,3))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=combined_loss,
              metrics=[psnr_metric, ssim_metric])
print("[info] compiled ok.")

CKPT_DIR = "/home/knarwani/thesis/super-resolution/models"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "best_by_val_loss.keras")

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10,
                           restore_best_weights=True, verbose=1)
model_ckpt = ModelCheckpoint(filepath=CKPT_PATH, monitor="val_loss",
                             mode="min", save_best_only=True, verbose=1)

print("[info] starting training…")
model.summary()
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=valid_ds,
                    callbacks=[early_stop, model_ckpt],
                    verbose=2)
print("[info] training done.")

# =========================
# 6) QUICK VAL METRICS
# =========================
EVAL_BATCH = 8
eval_ds = (tf.data.Dataset.from_tensor_slices((validation_low_image, validation_high_image))
           .batch(EVAL_BATCH).prefetch(tf.data.AUTOTUNE))

all_psnr, all_ssim, all_msssim = [], [], []; n_images = 0
for lr_b, hr_b in eval_ds:
    pred_b = model(lr_b, training=False)
    hr_tf   = tf.cast(hr_b, tf.float32)
    pred_tf = tf.cast(tf.clip_by_value(pred_b, 0.0, 1.0), tf.float32)
    all_psnr.append(tf.image.psnr(hr_tf, pred_tf, max_val=1.0).numpy())
    all_ssim.append(tf.image.ssim(hr_tf, pred_tf, max_val=1.0).numpy())
    all_msssim.append(tf.image.ssim_multiscale(hr_tf, pred_tf, max_val=1.0).numpy())
    n_images += int(hr_b.shape[0])

def mean_std(x):
    x = np.concatenate(x, axis=0).astype(np.float64)
    return float(np.mean(x)), float(np.std(x))

m_psnr, s_psnr     = mean_std(all_psnr)
m_ssim, s_ssim     = mean_std(all_ssim)
m_msssim, s_msssim = mean_std(all_msssim)
print(f"Validation images evaluated: {n_images}")
print(f" PSNR    : {m_psnr:.4f} ± {s_psnr:.4f} dB")
print(f" SSIM    : {m_ssim:.4f} ± {s_ssim:.4f}")
print(f" MS-SSIM : {m_msssim:.4f} ± {s_msssim:.4f}")