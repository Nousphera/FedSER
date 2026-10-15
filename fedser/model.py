"""Four-block spectral, temporal and channel attention model.

Channels-last execution supports CPU as well as GPU. Source mode preserves
the original axis ordering mathematically, including softmax over time.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="FedSER")
class STCAttention(tf.keras.layers.Layer):
    def __init__(self, mode="paper", **kwargs):
        super().__init__(**kwargs)
        if mode not in {"paper", "source"}:
            raise ValueError("mode must be paper or source")
        self.mode = mode

    def build(self, shape):
        _, time, frequency, channels = shape
        self.spectral = tf.keras.layers.Conv2D(1, (time, 1), padding="valid")
        self.temporal = tf.keras.layers.Conv2D(1, (1, frequency), padding="valid")
        self.dense1 = tf.keras.layers.Dense(channels // 2)
        self.dense2 = tf.keras.layers.Dense(channels, activation="tanh" if self.mode == "paper" else "sigmoid")
        super().build(shape)

    def call(self, x):
        if self.mode == "paper":
            pooled = tf.concat([tf.reduce_max(x, axis=-1, keepdims=True),
                                tf.reduce_mean(x, axis=-1, keepdims=True)], axis=-1)
        else:
            pooled = tf.concat([tf.nn.max_pool2d(x, 2, 1, "SAME"),
                                tf.nn.avg_pool2d(x, 2, 1, "SAME")], axis=-1)
        st = self.spectral(pooled) * self.temporal(pooled)
        channel = self.dense2(self.dense1(tf.reduce_mean(x, axis=(1, 2))))[:, None, None, :]
        return tf.nn.softmax(st * channel, axis=1)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def build_model(input_shape=(198, 64, 1), mode="paper", dropout=.1, l2=.0001):
    if len(input_shape) != 3 or min(input_shape[:2]) < 16 or input_shape[-1] != 1:
        raise ValueError("Expected (time>=16, mel>=16, 1) spectrogram shape")
    inputs = tf.keras.Input(shape=input_shape, name="log_mel")
    x = inputs
    for index, width in enumerate((24, 32, 64, 128)):
        branches = []
        for axis, kernel in (("spectral", (1, 4)), ("temporal", (4, 1))):
            y = tf.keras.layers.Conv2D(width, kernel, padding="same", kernel_regularizer=tf.keras.regularizers.L2(l2),
                                       name=f"block{index}_{axis}")(x)
            y = tf.keras.layers.GroupNormalization(groups=4, epsilon=1e-3)(y)
            branches.append(tf.keras.layers.ReLU()(y))
        x = tf.keras.layers.Concatenate()(branches)
        x = tf.keras.layers.Conv2D(width, 1, kernel_regularizer=tf.keras.regularizers.L2(l2))(x)
        x = tf.keras.layers.GroupNormalization(groups=4, epsilon=1e-3)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    scores = STCAttention(mode=mode, name="stc_attention")(x)
    weighted = tf.keras.layers.Multiply()([x, scores])
    x = tf.keras.layers.Add()([x, weighted])
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(4, name="emotion_logits")(x), name="fedser")
