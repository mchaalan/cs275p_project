import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Rescaling, Normalization
from tensorflow.keras.metrics import AUC

# Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 300
TRAIN_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/train'
VAL_DIR = '/Users/mohamadc/MI Dropbox Dropbox/Mohamad Chaalan/Mac/Downloads/chest_xray/test'

# Load datasets
train_ds = image_dataset_from_directory(
    TRAIN_DIR, label_mode='binary', batch_size=BATCH_SIZE,
    image_size=IMG_SIZE, shuffle=True
)
val_ds = image_dataset_from_directory(
    VAL_DIR, label_mode='binary', batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Normalize
normalizer = tf.keras.Sequential([
    Rescaling(1./255),
    Normalization()
])
normalizer.layers[1].adapt(train_ds.map(lambda x, y: x))
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (normalizer(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalizer(x), y)).prefetch(AUTOTUNE)

# Load EfficientNetV2 base
base_model = EfficientNetV2B3(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_model.trainable = True  # We will selectively freeze layers

# Freeze lower layers and fine-tune upper layers
for layer in base_model.layers[:-7]:
    layer.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile with recommended learning rate and metrics
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

# Optional: Load weights if continuing training
# model.load_weights("densenet_cnn.weights.h5")

# Train with early stopping and learning rate scheduler
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save_weights("densenet201_finetuned_pneumonia.h5")
