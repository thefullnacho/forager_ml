import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import os

# 1. Setup
os.environ["TF_USE_LEGACY_KERAS"] = "1"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 2. Augmentation (Hardening for Handheld Use)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# 3. Data Loader
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# Apply processing
train_ds = train_ds.map(lambda x, y: (tf.cast(data_augmentation(x), tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# 4. Build Model (MobileNetV3 Large - Clean Architecture)
inputs = Input(shape=(224, 224, 3), name="input_1")
base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs, include_top=False, weights='imagenet', include_preprocessing=False
)

# Deeper Unfreezing: Unfreeze top 60 layers for the hardened dataset
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.4)(x) 
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 5. Optimization
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- REFINED CLASS WEIGHTS ---
# Since the dataset is now balanced, we use weights for safety, not for count correction.
class_weight = {
    0: 1.5, # Deadly 
    1: 1.8, # Edible (Highest safety priority)
    2: 1.0, # Medicinal
    3: 0.8  # Plants (Slightly de-weighted to avoid the 143 false alarms)
}

# 6. Callbacks
checkpoint = callbacks.ModelCheckpoint(
    'forager_final_candidate.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

# 7. Train
model.fit(train_ds, validation_data=val_ds, epochs=30, class_weight=class_weight, callbacks=[checkpoint, lr_reducer, early_stop])
