import tensorflow as tf

image_size = 64
clip_min = -1.0
clip_max = 1.0
dataset_repetitions = 5

def augment(img):
    """Flips an image left/right randomly."""
    return tf.image.random_flip_left_right(img)

def resize_and_rescale(img, size):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [-1.0, 1.0].

    Args:
        img: Image tensor
        size: Desired image size for resizing
    Returns:
        Resized and rescaled image tensor
    """

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)

    img = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # Resize
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)

    # Rescale the pixel values
    img = img / 127.5 - 1.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img


def preprocess_image(img):
    img = tf.image.convert_image_dtype(img, tf.float32)  # ensure float32 in [0,1]
    img = resize_and_rescale(img, size=(image_size, image_size))
    img = augment(img)
    return img


def load_ddpm_dataset(dataset_path, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode=None,  # No labels needed for generative tasks
        image_size=(image_size, image_size),  # Resize here or later
        batch_size=None,  # So we can apply our own batching and augmentations
        shuffle=True
    )

    return (
        ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .shuffle(batch_size * 2)
        .prefetch(tf.data.AUTOTUNE)
    )

