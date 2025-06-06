import keras.optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization, Rescaling,
                                     RandomFlip, RandomRotation, RandomZoom)

# Config
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
EPOCHS = 50
TRAIN_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/train'
VAL_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/test'

# Load datasets
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

AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation layer
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# Apply augmentation only to training
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (Rescaling(1./255)(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (Rescaling(1./255)(x), y)).prefetch(AUTOTUNE)

# Model definition
model = Sequential([
    Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.2),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.3),

    # Block 4
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.4),

    # Block 5
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.5),

    # Block 6 — deeper abstract features
    Conv2D(1024, (3, 3), activation='relu', padding='same'),
    Conv2D(1024, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.5),

    # Classification Head
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)
