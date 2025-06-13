import tensorflow as tf
from tensorflow import keras
from keras import layers


@keras.utils.register_keras_serializable()
class KID(keras.metrics.Metric):
    def __init__(self, name, img_size, **kwargs):
        super().__init__(name=name, **kwargs)

        #KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        #a pretrained InceptionV3 is used without its classification layer
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(img_size, img_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=75, width=75),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(75, 75, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder_kid",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (tf.matmul(features_1, features_2, transpose_b=True) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        #compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        #estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # pdate the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


class DynamicKID:
    """class that computes Kernel Inception Distance (KID) for
    diffusion models trained with either the DDIM or DDPM objective.
    """

    def __init__(
        self,
        diffusion_type: str,
        img_size: int,
        kid_diffusion_steps: int = 5,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        name: str = "kid",
    ) -> None:
        if diffusion_type not in {"ddim", "ddpm"}:
            raise ValueError("diffusion_type must be either 'ddim' or 'ddpm'.")

        self.name = name
        self.diffusion_type = diffusion_type
        self.kid_diffusion_steps = kid_diffusion_steps
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.metric = KID(name=name, img_size=img_size)

    def result(self):
        """Return the current KID estimate."""
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()

    # update state for either ddim or ddpm models
    def update_state(self, real_images: tf.Tensor, model):
        """Update the internal KID metric given a batch of *real* images."""
        batch_size = tf.shape(real_images)[0]
        #prepare the generated images depending on the model type
        if self.diffusion_type == "ddim":
            #DDIM models keep images in [0, 1] range already
            generated_images = model.generate(
                num_images=batch_size,
                diffusion_steps=self.kid_diffusion_steps,
            )
            real_images_proc = real_images  # already in [0, 1] range
        else:  # DDPM 
            generated_images = model.generate(num_images=batch_size)
            #DDPM code works with a symmetric range [-1, 1]
            #convert both real and generated images to the [0, 1] range expected by KID
            real_images_proc = self._to_unit_range(real_images)
            generated_images = self._to_unit_range(generated_images)

        #update the underlying
        self.metric.update_state(real_images_proc, generated_images)

    #helper
    def _to_unit_range(self, imgs: tf.Tensor) -> tf.Tensor:
        """Rescale tensors from [clip_min, clip_max] to [0, 1]."""
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        return tf.clip_by_value(imgs, 0.0, 1.0)


#Frechet Inception Distance (FID)

@keras.utils.register_keras_serializable()
class FID(keras.metrics.Metric):
    """Fréchet Inception Distance metric"""

    def __init__(self, name="fid", feature_dim: int = 2048, **kwargs):
        super().__init__(name=name, **kwargs)
        self.d = feature_dim

        #running sums for real images
        self.real_n = self.add_weight(shape=(), initializer="zeros", dtype=tf.float64)
        self.real_sum = self.add_weight(shape=(feature_dim,), initializer="zeros", dtype=tf.float64)
        self.real_outer = self.add_weight(shape=(feature_dim, feature_dim), initializer="zeros", dtype=tf.float64)

        #running sums for fake images
        self.fake_n = self.add_weight(shape=(), initializer="zeros", dtype=tf.float64)
        self.fake_sum = self.add_weight(shape=(feature_dim,), initializer="zeros", dtype=tf.float64)
        self.fake_outer = self.add_weight(shape=(feature_dim, feature_dim), initializer="zeros", dtype=tf.float64)

        #Inception encoder (pooling="avg" gives 2048-D activations)
        self.encoder = keras.Sequential(
            [
                layers.Resizing(299, 299),
                layers.Rescaling(255.0),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling="avg"),
            ],
            name="inception_encoder_fid",
        )

    def _update_sums(self, features, real: bool):
        features = tf.cast(features, tf.float64)
        batch_n = tf.shape(features)[0]
        batch_sum = tf.reduce_sum(features, axis=0)
        batch_outer = tf.matmul(features, features, transpose_a=True)
        if real:
            self.real_n.assign_add(tf.cast(batch_n, tf.float64))
            self.real_sum.assign_add(batch_sum)
            self.real_outer.assign_add(batch_outer)
        else:
            self.fake_n.assign_add(tf.cast(batch_n, tf.float64))
            self.fake_sum.assign_add(batch_sum)
            self.fake_outer.assign_add(batch_outer)

    def update_state(self, imgs: tf.Tensor, real: bool):
        feats = self.encoder(imgs, training=False)
        self._update_sums(feats, real)

    def _mean_and_cov(self, n, s1, s2):
        mean = s1 / n
        cov = s2 / (n - 1.0) - tf.tensordot(mean, mean, axes=0) * n / (n - 1.0)
        return mean, cov

    def result(self):
        #compute mean and covariance for real and fake activations
        mu_r, sigma_r = self._mean_and_cov(self.real_n, self.real_sum, self.real_outer)
        mu_f, sigma_f = self._mean_and_cov(self.fake_n, self.fake_sum, self.fake_outer)

        mu_diff = mu_r - mu_f
        cov_mean = tf.linalg.sqrtm(tf.cast(tf.matmul(sigma_r, sigma_f), tf.complex128))
        cov_mean = tf.math.real(cov_mean)
        fid = tf.tensordot(mu_diff, mu_diff, axes=1) + tf.linalg.trace(sigma_r + sigma_f - 2.0 * cov_mean)
        return tf.cast(fid, tf.float32)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


class DynamicFID:
    """Wrapper that mirrors DynamicKID but for Fréchet Inception Distance."""
    def __init__(
        self,
        diffusion_type: str,
        fid_diffusion_steps: int = 20,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        name: str = "fid",
    ):
        self.name = name
        self.diffusion_type = diffusion_type
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.fid_diffusion_steps = fid_diffusion_steps
        self.metric = FID(name=name)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()

    def _to_unit_range(self, x):
        x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        return tf.clip_by_value(x, 0.0, 1.0)

    def update_state(self, real_images: tf.Tensor, model):
        batch_size = tf.shape(real_images)[0]
        if self.diffusion_type == "ddim":
            generated_images = model.generate(batch_size, diffusion_steps=self.fid_diffusion_steps)
            real_images_proc = real_images
        else:  # DDPM
            generated_images = model.generate(batch_size)
            generated_images = self._to_unit_range(generated_images)
            real_images_proc = self._to_unit_range(real_images)

        self.metric.update_state(real_images_proc, real=True)
        self.metric.update_state(generated_images, real=False)


#  TODO: CLIP-BASED MMD (CLIP-MMD) METRIC
class CLIPMMD(keras.metrics.Metric):
    pass