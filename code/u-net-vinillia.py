x#!/usr/bin/env python
# coding: utf-8

# ![Screenshot 2021-11-27 at 4.02.39 PM.png](attachment:130a46f2-9046-4106-b324-1983094c922f.png)

# In[ ]:


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

# In[26]:


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

# In[31]:


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
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    # correctly upsample from the `inputs` tensor, not undefined `x`
    x = UpSampling2D()(inputs)
    x = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_super_resolution_unet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs,  64)   # 256 → 128
    s2, p2 = encoder_block(p1,        128) # 128 → 64
    s3, p3 = encoder_block(p2,        256) # 64  → 32
    s4, p4 = encoder_block(p3,        512) # 32  → 16

    # Bridge
    b1 = conv_block(p4, 1024)               # 16

    # Decoder (back up to 256×256)
    d1 = decoder_block(b1,  s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1,  64)




    # ─── Extra upsampling for true 2× SR ───
    u1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 256 → 512
    u1 = Conv2D(64, 3, padding="same", activation="relu")(u1)
    u1 = conv_block(u1, 64)

    x = Conv2D(3, 1, padding="same")(d4)
    outputs = Activation("sigmoid")(x)

    return Model(inputs, outputs, name="U-Net_SR")

# Quick check
if __name__ == "__main__":
    model = build_super_resolution_unet((256, 256, 3))
    plot_model(model, to_file="unet_sr_2x.png", show_shapes=True)


# In[37]:


# ===== Losses & Metrics  =====

# --- losses ---
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_metric(y_true, y_pred):
    # assumes inputs in [0,1]
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# --- VGG feature extractor (float32 by default) ---
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
feat_extractor = Model(
    inputs=vgg.input,
    outputs=vgg.get_layer('block5_conv4').output
)


# KEY CHANGE: clip→cast→VGG19.preprocess_input (does RGB->BGR + mean subtraction)
def perceptual_loss(y_true, y_pred):
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    yt = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
    yp = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)
    ft = feat_extractor(yt)
    fp = feat_extractor(yp)
    return tf.reduce_mean(tf.square(ft - fp))

# Recommend starting weights: strong L1/MSE, small SSIM, tiny perceptual
α, β, γ = 1.0, 0.1, 0.01

def combined_loss(y_true, y_pred):
    return (α * mse_loss(y_true, y_pred)
          + β * ssim_loss(y_true, y_pred)
          + γ * perceptual_loss(y_true, y_pred))
# 4) (Optional) track SSIM as a metric too
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# 5) Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=[psnr_metric, ssim_metric]
)


# In[ ]:


# ─── Dataset from in-memory arrays ─────────────────────────────────────────────
BATCH_SIZE = 4   # pick what fits your GPU/CPU
EPOCHS = 100

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_low_image, train_high_image))
      .shuffle(len(train_low_image))
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((validation_low_image, validation_high_image))
      .batch(BATCH_SIZE)
      .prefetch(tf.data.AUTOTUNE)
)

# ─── Callbacks ────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

model_ckpt = ModelCheckpoint(
    filepath="/models/best_by_val_loss.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
)

# ─── Training ────────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[early_stop, model_ckpt],
    verbose=2,
)


# In[36]:


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

