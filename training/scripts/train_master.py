import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import os

# 1. Setup
os.environ["TF_USE_LEGACY_KERAS"] = "1"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 2. Advanced Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2), # Increased for varying woodland light
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

# Normalize Mapping
def process_train(img, label):
    return tf.cast(data_augmentation(img), tf.float32) / 255.0, label

def process_val(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(process_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(process_val, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# 4. Build Architecture (MobileNetV3 Large)
inputs = Input(shape=(224, 224, 3), name="input_1")
base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs, include_top=False, weights='imagenet', include_preprocessing=False
)

# Initial Fine-Tuning: Unfreeze top 50 layers for deeper feature extraction
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.4)(x) # Higher dropout to fight that 95% overfit
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 5. Optimization Strategy
# Using a slightly higher learning rate with a scheduler to refine later
model.compile(optimizer=tf.keras.optimizers.Adam(2e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- CRITICAL: CLASS WEIGHTING ---
# We prioritize Edible and Deadly classes to fix the 'Kill Zone' errors
#
class_weight = {
    0: 1.8, # Deadly (Actual index 0)
    1: 2.5, # Edible (Actual index 1) - Highest weight to prevent false danger
    2: 1.2, # Medicinal (Actual index 2)
    3: 1.0  # Plants (Actual index 3)
}

# 6. Advanced Callbacks
checkpoint = callbacks.ModelCheckpoint(
    'forager_master_v1.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)

# Automatically drops learning rate when progress stalls
lr_reducer = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=8, restore_best_weights=True
)

# 7. Start the Master Run
print("🚀 Starting Master Training Run...")
model.fit(
    train_ds, validation_data=val_ds, epochs=40, # More epochs, scheduler will handle the end
    class_weight=class_weight,
    callbacks=[checkpoint, lr_reducer, early_stop]
)
