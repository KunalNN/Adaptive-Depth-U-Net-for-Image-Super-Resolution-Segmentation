from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import layers as L


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


@register_keras_serializable(package="resize")
class ResizeByScale(L.Layer):
    def __init__(self, scale: float, method: str = "bilinear", antialias: bool = True, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.method = method
        self.antialias = antialias

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_dtype = x.dtype  # preserve incoming dtype
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        nh = tf.cast(tf.math.ceil(tf.cast(h, tf.float32) * self.scale), tf.int32)
        nw = tf.cast(tf.math.ceil(tf.cast(w, tf.float32) * self.scale), tf.int32)
        nh = tf.maximum(nh, 1)
        nw = tf.maximum(nw, 1)

        resized = tf.image.resize(tf.cast(x, tf.float32), [nh, nw], method=self.method, antialias=self.antialias)
        return tf.cast(resized, x_dtype)

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
        x, ref = inputs
        target_hw = tf.shape(ref)[1:3]
        resized = tf.image.resize(tf.cast(x, tf.float32), target_hw, method=self.method, antialias=self.antialias)
        return tf.cast(resized, x.dtype)

    def get_config(self) -> Dict[str, object]:
        return {
            **super().get_config(),
            "method": self.method,
            "antialias": self.antialias,
        }

@register_keras_serializable(package="utils")
class ClipAdd(L.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        inp, residual = inputs
        out = tf.cast(inp, tf.float32) + tf.cast(residual, tf.float32)
        return tf.cast(tf.clip_by_value(out, 0.0, 1.0), inp.dtype)
