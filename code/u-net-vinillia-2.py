#!/usr/bin/env python
# coding: utf-8

# ![Screenshot 2021-11-27 at 4.02.39 PM.png](attachment:130a46f2-9046-4106-b324-1983094c922f.png)

# In[1]:


import os 
import re 
from scipy import ndimage, misc 
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array


from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import numpy as np
np. random. seed(0)
import cv2 as cv2

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense ,Conv2D,MaxPooling2D ,Dropout, Activation,BatchNormalization,MaxPool2D,Concatenate,Flatten, Lambda
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.applications import VGG19

print(tf.__version__)


# ![Screenshot 2021-11-27 at 4.04.31 PM.png](attachment:8dcd3305-b356-4f8e-ba85-662cf89fbe62.png)

# In[2]:


# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)
# defining the size of the image
SIZE = 256
high_img = []
path = '/Users/kunalnarwani/Desktop/Thesis/super-resolution/dataset/Raw Data/high_res' 
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):    
    if i == '855.png':
        break
    else:    
        img = cv2.imread(path + '/'+i,1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        high_img.append(img_to_array(img))


low_img = []
path = '/Users/kunalnarwani/Desktop/Thesis/super-resolution/dataset/Raw Data/low_res'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):
    if i == '855.png':
        break
    else: 
        img = cv2.imread(path + '/'+i,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        low_img.append(img_to_array(img))

train_high_image = high_img[:700]
train_low_image = low_img[:700]
train_high_image = np.reshape(train_high_image,(len(train_high_image),SIZE,SIZE,3))
train_low_image = np.reshape(train_low_image,(len(train_low_image),SIZE,SIZE,3))

validation_high_image = high_img[700:810]
validation_low_image = low_img[700:810]
validation_high_image= np.reshape(validation_high_image,(len(validation_high_image),SIZE,SIZE,3))
validation_low_image = np.reshape(validation_low_image,(len(validation_low_image),SIZE,SIZE,3))


test_high_image = high_img[810:]
test_low_image = low_img[810:]
test_high_image= np.reshape(test_high_image,(len(test_high_image),SIZE,SIZE,3))
test_low_image = np.reshape(test_low_image,(len(test_low_image),SIZE,SIZE,3))

print("Shape of training images:",train_high_image.shape)
print("Shape of test images:",test_high_image.shape)
print("Shape of validation images:",validation_high_image.shape)


# # ARCITECTURE OF MODEL 
# 

# In[ ]:


# ---------------- Blocks ---------------- #
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D(pool_size=(2, 2))(x)   # /2
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(inputs)  # ×2
    x = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# ------------- Model (256 -> 256) ------------- #
def build_super_resolution_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder: 256 -> 128 -> 64 -> 32 -> 16
    s1, p1 = encoder_block(inputs,   64)   # 256
    s2, p2 = encoder_block(p1,      128)   # 128
    s3, p3 = encoder_block(p2,      256)   # 64
    s4, p4 = encoder_block(p3,      512)   # 32

    # Bridge at 16×16
    b1 = conv_block(p4, 1024)

    # Decoder: 16 -> 32 -> 64 -> 128 -> 256
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1,  64)  # back to 256×256 here

    # ---- SAME-SIZE HEAD (no upscaling) ----
    # Recommended: residual enhancement (keeps identity at init)
    # res = Conv2D(3, 1, padding="same",
    #              kernel_initializer="zeros", bias_initializer="zeros",
    #              name="residual_rgb")(d4)
    # outputs = Lambda(
    #     lambda t: tf.clip_by_value(tf.cast(t[0], tf.float32) + tf.cast(t[1], tf.float32), 0.0, 1.0),
    #     name="enhanced_rgb"
    # )([inputs, res])

    outputs = Activation("sigmoid")(Conv2D(3, 1, padding="same")(d4))

    return Model(inputs, outputs, name="U-Net_SR_256x256_same_size")

# Quick check
if __name__ == "__main__":
    model = build_super_resolution_unet((256, 256, 3))
    y = model(tf.zeros([1, 256, 256, 3]))
    print("Output shape:", y.shape)  # (1, 256, 256, 3)


# In[4]:


from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

vgg = VGG19(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
vgg.trainable = False
feat_extractor = tf.keras.Model(
    inputs=vgg.input,
    outputs=vgg.get_layer("block4_conv4").output,
)


# In[5]:


alpha = tf.constant(1.0,  dtype=tf.float32)
beta  = tf.constant(0.1,  dtype=tf.float32)
gamma = tf.constant(0.01, dtype=tf.float32)

def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    val = tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.cast(val, tf.float32)

def ssim_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    val = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return tf.cast(val, tf.float32)

def perceptual_loss(y_true, y_pred):
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    feat_true = feat_extractor(preprocess_input(y_true * 255.0))
    feat_pred = feat_extractor(preprocess_input(y_pred * 255.0))
    val = tf.reduce_mean(tf.square(feat_true - feat_pred))
    return tf.cast(val, tf.float32)

def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    total = (
        alpha * mse_loss(y_true, y_pred)
        + beta  * ssim_loss(y_true, y_pred)
        + gamma * perceptual_loss(y_true, y_pred)
    )
    return tf.cast(total, tf.float32)


# In[6]:


# ===== Compile =====

def psnr_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=[psnr_metric],
)


# In[7]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, BackupAndRestore

BATCH_SIZE = 4
EPOCHS = 100
SEED = 42  # optional, keeps shuffle deterministic across restarts

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_low_image, train_high_image))
      .shuffle(len(train_low_image), seed=SEED, reshuffle_each_iteration=True)
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((validation_low_image, validation_high_image))
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

MODEL_DIR = "/Users/kunalnarwani/Desktop/Thesis/super-resolution/models"
os.makedirs(MODEL_DIR, exist_ok=True)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

model_ckpt = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_by_val_loss.keras"),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

backup_cb = BackupAndRestore(os.path.join(MODEL_DIR, "train_backup"))

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[early_stop, model_ckpt, backup_cb],
    verbose=2,
)


# In[ ]:


EVAL_BATCH = 8  # adjust for GPU/CPU memory
eval_ds = (
    tf.data.Dataset.from_tensor_slices((validation_low_image, validation_high_image))
      .batch(EVAL_BATCH)
      .prefetch(tf.data.AUTOTUNE)
)

all_psnr, all_ssim, all_msssim = [], [], []
n_images = 0

for lr_b, hr_b in eval_ds:
    pred_b = model(lr_b, training=False)

    if pred_b.shape[1:3] != hr_b.shape[1:3]:  # still guards future changes
        pred_b = tf.image.resize(pred_b, size=hr_b.shape[1:3], method="bicubic")

    hr_tf   = tf.cast(hr_b, tf.float32)
    pred_tf = tf.cast(tf.clip_by_value(pred_b, 0.0, 1.0), tf.float32)

    all_psnr.append(tf.image.psnr(hr_tf, pred_tf, max_val=1.0).numpy())
    all_ssim.append(tf.image.ssim(hr_tf, pred_tf, max_val=1.0).numpy())
    all_msssim.append(tf.image.ssim_multiscale(hr_tf, pred_tf, max_val=1.0).numpy())

    n_images += int(hr_b.shape[0])

def mean_std(x):
    x = np.concatenate(x, axis=0).astype(np.float64)
    return float(np.mean(x)), float(np.std(x))

m_psnr, s_psnr   = mean_std(all_psnr)
m_ssim, s_ssim   = mean_std(all_ssim)
m_msssim, s_msssim = mean_std(all_msssim)

print(f"Validation images evaluated: {n_images}")
print(f" PSNR    : {m_psnr:.4f} ± {s_psnr:.4f} dB")
print(f" SSIM    : {m_ssim:.4f} ± {s_ssim:.4f}")
print(f" MS-SSIM : {m_msssim:.4f} ± {s_msssim:.4f}")

