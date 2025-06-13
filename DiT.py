import keras
from keras import layers
from keras import ops
import math


@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x, embedding_dims=128):
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
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=-1
    )
    return embeddings


@keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=384, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid'
        )

    def call(self, x):
        # x shape: (batch, height, width, channels)
        x = self.projection(x)  # (batch, h//patch_size, w//patch_size, embed_dim)
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        x = ops.reshape(x, (batch_size, height * width, self.embed_dim))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.saving.register_keras_serializable()
class DiTBlock(layers.Layer):
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.0
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])

        # AdaLN-Zero modulation
        self.adaLN_modulation = layers.Dense(6 * embed_dim, kernel_initializer='zeros')

    def call(self, x, c):
        # x: (batch, seq_len, embed_dim)
        # c: (batch, embed_dim) - conditioning vector

        # AdaLN-Zero modulation
        mod = self.adaLN_modulation(c)  # (batch, 6 * embed_dim)
        mod = ops.reshape(mod, (-1, 6, self.embed_dim))  # (batch, 6, embed_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.unstack(mod, axis=1)

        # Self-attention with AdaLN-Zero
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + ops.expand_dims(scale_msa, 1)) + ops.expand_dims(shift_msa, 1)
        attn_out = self.attn(norm_x, norm_x)
        x = x + ops.expand_dims(gate_msa, 1) * attn_out

        # MLP with AdaLN-Zero
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + ops.expand_dims(scale_mlp, 1)) + ops.expand_dims(shift_mlp, 1)
        mlp_out = self.mlp(norm_x)
        x = x + ops.expand_dims(gate_mlp, 1) * mlp_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        })
        return config


"""
The final layer converts token embeddings into patch pixel predictions. 
Then the unpatchify layer stitches them into a full image.
"""
@keras.saving.register_keras_serializable()
class FinalLayer(layers.Layer):
    def __init__(self, patch_size=4, out_channels=3, embed_dim=384, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.norm_final = layers.LayerNormalization(epsilon=1e-6)
        self.linear = layers.Dense(patch_size * patch_size * out_channels, kernel_initializer='zeros')
        self.adaLN_modulation = layers.Dense(2 * embed_dim, kernel_initializer='zeros')

    def call(self, x, c):
        # x: (batch, seq_len, embed_dim)
        # c: (batch, embed_dim)

        mod = self.adaLN_modulation(c)  # (batch, 2 * embed_dim)
        shift, scale = ops.split(mod, 2, axis=-1)  # Each (batch, embed_dim)

        x = self.norm_final(x)
        x = x * (1 + ops.expand_dims(scale, 1)) + ops.expand_dims(shift, 1)
        x = self.linear(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "out_channels": self.out_channels,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_embed = layers.Embedding(num_patches, embed_dim)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        positions = ops.arange(self.num_patches)
        positions = ops.expand_dims(positions, 0)  # (1, num_patches)
        positions = ops.broadcast_to(positions, (batch_size, self.num_patches))
        pos_emb = self.pos_embed(positions)
        return x + pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return config


"""
This class when called, takes the linear outputs for each patch and reassembles 
them into a full image.
"""
@keras.saving.register_keras_serializable()
class UnpatchifyLayer(layers.Layer):
    def __init__(self, img_size, patch_size, out_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.h_patches = img_size // patch_size
        self.w_patches = img_size // patch_size

    def call(self, x):
        # x shape: (batch, num_patches, patch_size^2 * out_channels)
        batch_size = ops.shape(x)[0]

        # Reshape to (batch, h_patches, w_patches, patch_size, patch_size, out_channels)
        x = ops.reshape(x, (batch_size, self.h_patches, self.w_patches,
                            self.patch_size, self.patch_size, self.out_channels))

        # Rearrange to (batch, h_patches * patch_size, w_patches * patch_size, out_channels)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, self.img_size, self.img_size, self.out_channels))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "out_channels": self.out_channels,
        })
        return config


def dit_model(img_size=64, patch_size=8, embed_dim=256, depth=6, num_heads=4):
    noisy_images = keras.Input(shape=(img_size, img_size, 3), name="noisy_images")
    noise_variances = keras.Input(shape=(1, 1, 1), name="noise_variances")

    # Time embedding
    t_emb = ops.squeeze(noise_variances, axis=[1, 2])  # (batch, 1)
    t_emb = sinusoidal_embedding(t_emb, embedding_dims=embed_dim)  # (batch, embed_dim)

    # Patch embedding
    patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
    x = patch_embed(noisy_images)  # (batch, num_patches, embed_dim)

    # Positional embedding
    num_patches = (img_size // patch_size) ** 2
    pos_embed = PositionalEmbedding(num_patches=num_patches, embed_dim=embed_dim)
    x = pos_embed(x)

    # DiT blocks
    for _ in range(depth):
        dit_block = DiTBlock(embed_dim=embed_dim, num_heads=num_heads)
        x = dit_block(x, t_emb)

    # Final layer
    final_layer = FinalLayer(patch_size=patch_size, out_channels=3, embed_dim=embed_dim)
    x = final_layer(x, t_emb)

    unpatchify = UnpatchifyLayer(img_size=img_size, patch_size=patch_size, out_channels=3)
    x = unpatchify(x)

    return keras.Model([noisy_images, noise_variances], x, name="dit_model")