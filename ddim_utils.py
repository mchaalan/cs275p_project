import tensorflow as tf
from keras import ops

image_size = 64
dataset_repetitions = 5

def preprocess_img(data):
    # center crop image
    img = tf.image.convert_image_dtype(data, tf.float32)
    height = ops.shape(img)[0]
    width = ops.shape(img)[1]
    crop_size = ops.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return ops.clip(image / 255.0, 0.0, 1.0)


def load_ddim_dataset(dataset_path, batch_size):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            label_mode=None,  # No labels needed for generative tasks
            image_size=(image_size, image_size),  # Resize here or later
            batch_size=None,  # So we can apply our own batching and augmentations
            shuffle=True
        ).map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )