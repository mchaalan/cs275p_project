import keras
from keras import layers
from keras import ops
import math
import tensorflow as tf


@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_dims = 128  # Increased for richer representations
    embedding_max_frequency = 1000.0
    embedding_min_frequency = 1.0
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def get_group_count(channels):
    # Common divisors that work well
    possible_groups = [1, 2, 4, 8, 16, 32]
    for groups in reversed(possible_groups):
        if channels % groups == 0 and groups <= channels:
            return groups
    return 1  # fallback


class AdaptiveFeatureModulation(layers.Layer):
    """Modulates features based on noise level - revolutionary for diffusion!"""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.scale_net = keras.Sequential([
            layers.Dense(self.channels * 2, activation='swish'),
            layers.Dense(self.channels * 2, activation='swish'),
            layers.Dense(self.channels * 2)
        ])

    def call(self, x, time_emb):
        # x: feature maps, time_emb: time embedding
        scale_shift = self.scale_net(time_emb)
        scale_shift = layers.Reshape((1, 1, self.channels * 2))(scale_shift)
        scale, shift = tf.split(scale_shift, 2, axis=-1)
        return x * (1 + scale) + shift


class CrossScaleAttention(layers.Layer):
    def __init__(self, channels, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = min(num_heads, channels // 16)
        if self.num_heads <= 0:
            self.num_heads = 1
        self.head_dim = channels // self.num_heads

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.channels)
        self.k_proj = layers.Dense(self.channels)
        self.v_proj = layers.Dense(self.channels)
        self.out_proj = layers.Dense(self.channels)

        self.multi_scale_proj = layers.Conv2D(self.channels, 1)

        # Initialize MultiHeadAttention once
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            dropout=0.1
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        height, width = x.shape[1], x.shape[2]

        x_1x1 = x
        x_2x2 = layers.AveragePooling2D(2, padding='same')(x)
        x_4x4 = layers.AveragePooling2D(4, padding='same')(x)

        x_2x2 = layers.UpSampling2D(2, interpolation='bilinear')(x_2x2)
        x_4x4 = layers.UpSampling2D(4, interpolation='bilinear')(x_4x4)

        multi_scale = layers.Concatenate()([x_1x1, x_2x2, x_4x4])

        multi_scale = self.multi_scale_proj(multi_scale)

        q = self.q_proj(x)
        k = self.k_proj(multi_scale)
        v = self.v_proj(multi_scale)

        seq_len = height * width
        q_flat = tf.reshape(q, [batch_size, seq_len, self.channels])
        k_flat = tf.reshape(k, [batch_size, seq_len, self.channels])
        v_flat = tf.reshape(v, [batch_size, seq_len, self.channels])

        attended = self.attention_layer(q_flat, k_flat, v_flat)
        attended = tf.reshape(attended, [batch_size, height, width, self.channels])

        output = self.out_proj(attended)
        return output + x


def EnhancedResidualBlock(width, use_attention=True):
    def apply(x, time_emb=None):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)

        # Fixed Group normalization
        groups = get_group_count(width)
        x = layers.GroupNormalization(groups=groups)(x)
        x = layers.Activation("swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        # Adaptive feature modulation with time embedding
        if time_emb is not None:
            afm = AdaptiveFeatureModulation(width)
            x = afm(x, time_emb)

        groups = get_group_count(width)
        x = layers.GroupNormalization(groups=groups)(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(0.1)(x)  # Slight regularization
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        # Cross-scale attention - only for larger feature maps and channels
        if use_attention and width >= 128 and x.shape[1] >= 8:  # Ensure reasonable size
            x = CrossScaleAttention(width)(x)

        x = layers.Add()([x, residual])
        return x

    return apply


class ProgressiveFeatureFusion(layers.Layer):
    """Progressively fuses features from different scales"""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.fusion_conv = layers.Conv2D(self.channels, 3, padding='same')
        self.attention_weights = layers.Conv2D(1, 1, activation='sigmoid')

    def call(self, current_features, skip_features):
        # Ensure same spatial dimensions
        if current_features.shape[1:3] != skip_features.shape[1:3]:
            skip_features = layers.Resizing(
                current_features.shape[1],
                current_features.shape[2]
            )(skip_features)

        # Ensure same channel dimensions
        if current_features.shape[-1] != skip_features.shape[-1]:
            skip_features = layers.Conv2D(
                current_features.shape[-1], 1
            )(skip_features)

        # Adaptive fusion based on content
        combined = layers.Concatenate()([current_features, skip_features])
        fused = self.fusion_conv(combined)

        # Learn fusion weights
        weights = self.attention_weights(combined)

        return weights * current_features + (1 - weights) * skip_features + fused * 0.1


def AdvancedDownBlock(width, block_depth, use_attention=False):
    def apply(inputs):
        if isinstance(inputs, list) and len(inputs) == 2:
            x, skips = inputs[0], inputs[1]
            time_emb = None
        elif isinstance(inputs, list) and len(inputs) == 3:
            x, skips, time_emb = inputs[0], inputs[1], inputs[2]
        else:
            x, skips = inputs
            time_emb = None

        for i in range(block_depth):
            use_attn = use_attention and i == block_depth - 1  # Attention on last layer
            x = EnhancedResidualBlock(width, use_attention=use_attn)(x, time_emb)
            skips.append(x)

        # Learnable downsampling
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same")(x)
        return x

    return apply


def AdvancedUpBlock(width, block_depth, use_attention=False):
    def apply(inputs):
        if isinstance(inputs, list) and len(inputs) == 2:
            x, skips = inputs[0], inputs[1]
            time_emb = None
        elif isinstance(inputs, list) and len(inputs) == 3:
            x, skips, time_emb = inputs[0], inputs[1], inputs[2]
        else:
            x, skips = inputs
            time_emb = None

        # Learnable upsampling
        x = layers.Conv2DTranspose(width, kernel_size=3, strides=2, padding="same")(x)

        for i in range(block_depth):
            if skips:  # Check if skips is not empty
                skip = skips.pop()

                # Progressive feature fusion instead of simple concatenation
                pff = ProgressiveFeatureFusion(width)
                x = pff(x, skip)

            use_attn = use_attention and i == 0  # Attention on first layer
            x = EnhancedResidualBlock(width, use_attention=use_attn)(x, time_emb)
        return x

    return apply


def revolutionary_unet_model(img_size, widths, block_depth):
    noisy_images = keras.Input(shape=(img_size, img_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    # Enhanced time embedding
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 128))(noise_variances)
    e = layers.UpSampling2D(size=img_size, interpolation="nearest")(e)

    # Global time embedding for modulation
    time_emb_global = layers.GlobalAveragePooling2D()(e)
    time_emb_global = layers.Dense(512, activation='swish')(time_emb_global)
    time_emb_global = layers.Dense(256, activation='swish')(time_emb_global)

    # Initial processing with richer features
    x = layers.Conv2D(widths[0], kernel_size=3, padding="same")(noisy_images)
    x = layers.Concatenate()([x, e])
    x = layers.Conv2D(widths[0], kernel_size=3, padding="same")(x)

    skips = []

    # Encoder with progressive attention
    for i, width in enumerate(widths[:-1]):
        use_attention = i >= len(widths) // 2  # Attention in deeper layers
        x = AdvancedDownBlock(width, block_depth, use_attention)([x, skips, time_emb_global])

    # Bottleneck with full attention
    for _ in range(block_depth):
        x = EnhancedResidualBlock(widths[-1], use_attention=True)(x, time_emb_global)

    # Decoder with progressive fusion
    for i, width in enumerate(reversed(widths[:-1])):
        use_attention = i < len(widths) // 2  # Attention in earlier decoder layers
        x = AdvancedUpBlock(width, block_depth, use_attention)([x, skips, time_emb_global])

    # Final output with residual connection
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="swish")(x)
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="revolutionary_unet")
