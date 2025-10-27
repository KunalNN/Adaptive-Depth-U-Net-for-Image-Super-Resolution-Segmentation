from __future__ import annotations

from typing import Dict, Tuple
from math import ceil
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


def depth_and_sizes(scale, min_res=21, max_depth=7):
    depth = 1
    sizes = [256]
    res = 256
    while res > min_res and depth < max_depth:
        res = ceil(res * scale)
        sizes.append(res)
        depth += 1
    depth = min(depth, max_depth)
    return depth, sizes

def custom_depth_from_scale(
    scale: float,
    min_depth: int = 1,
    max_depth: int = 7,
    *,
    base_resolution: int = 256,
    min_feature: int = 21,
) -> int:
    """
    Decide encoder depth by shrinking the spatial extent until it approaches the
    minimum feature size or the depth limit is reached.
    """
    if not (0.05 < scale < 1.0):
        raise ValueError("Scale should be between 0 and 1 (exclusive).")
    if min_depth < 1:
        raise ValueError("min_depth must be at least 1.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")
    if base_resolution <= 0:
        raise ValueError("base_resolution must be positive.")
    if min_feature < 1:
        raise ValueError("min_feature must be at least 1 pixel.")

    depth = max(min_depth, 1)
    feature_extent = base_resolution

    while feature_extent > min_feature and depth < max_depth:
        feature_extent = ceil(feature_extent * scale)
        depth += 1

    return max(min_depth, min(depth, max_depth))

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
