import tensorflow as tf
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.preprocessing import image_dataset_from_directory

base_model = tf.keras.applications.ResNet152V2(
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 300
TRAIN_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/train'
VAL_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/test'

# Load data
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

base_model.trainable = True

for layer in base_model.layers[:-10]:
    layer.trainable = False