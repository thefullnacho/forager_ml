import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import os

# 1. Setup
os.environ["TF_USE_LEGACY_KERAS"] = "1"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 2. Define Augmentation (The Gym for your Model)
# This forces the model to recognize mushrooms sideways, zoomed in, etc.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2), # Rotate up to 20%
    layers.RandomZoom(0.2),     # Zoom in/out up to 20%
    layers.RandomContrast(0.1), # Slight lighting changes
])

# 3. Data Loader
print("📂 Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 4. Processing Pipeline
# Train: Augment -> Normalize
def process_train(img, label):
    img = data_augmentation(img)
    return tf.cast(img, tf.float32) / 255.0, label

# Val: Normalize Only (Never augment validation data!)
def process_val(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(process_train, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_val, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetching for speed
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 5. Build Model
print("🏗️  Building Hardened Architecture...")
inputs = Input(shape=(224, 224, 3), name="input_1")

base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs,
    include_top=False,
    weights='imagenet',
    include_preprocessing=False
)

# Fine-Tuning: Unfreeze top 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
# Increased Dropout to 0.3 for extra resistance against overfitting
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Callbacks
checkpoint = callbacks.ModelCheckpoint(
    'forager_hardened.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5, # Stop if no improvement for 5 epochs
    restore_best_weights=True
)

# 7. Train (25 Epochs - Augmentation needs more time)
print("🚀 Starting Hardened Training...")
model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=25, 
    callbacks=[checkpoint, early_stop]
)

print("\n✅ Success! The BEST model was saved as 'forager_hardened.keras'.")
