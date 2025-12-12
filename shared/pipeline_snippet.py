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
