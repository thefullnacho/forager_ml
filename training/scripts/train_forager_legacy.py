import os
import tensorflow as tf
import numpy as np
import albumentations as A
from pathlib import Path

# --- 1. CONFIG ---
BASE = Path('inat_dataset')
CLASSES = ['deadly', 'edible', 'medicinal', 'plants']
IMG_SIZE = 224
BATCH_SIZE = 16 
EPOCHS = 15

# --- 2. DATA LOADING ---
def load_data():
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = BASE / cls
        if not folder.exists(): continue
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(entry.path)
                    labels.append(idx)
    return paths, labels

paths, labels = load_data()
print(f"✅ Total images found: {len(paths)}")

# --- 3. AUGMENTATION & DATASET ---
aug = A.Compose([
    A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
])

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    
    def augment_np(x):
        augmented = aug(image=(x.numpy() * 255).astype(np.uint8))
        return augmented["image"].astype(np.float32) / 255.0
    
    img = tf.cond(tf.random.uniform([]) < 0.8,
                  lambda: tf.py_function(augment_np, [img], tf.float32),
                  lambda: img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label

full_dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).shuffle(len(paths), seed=42)
val_size = int(len(paths) * 0.15)
val_ds = full_dataset.take(val_size).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds = full_dataset.skip(val_size).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 4. MODEL SETUP ---
base = tf.keras.applications.MobileNetV3Large(
    weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4), # Increased for better generalization
    tf.keras.layers.Dense(4, activation='softmax')
])

# --- 5. LEARNING RATE SCHEDULE (Your Strategy) ---
def lr_schedule(epoch):
    """Replicates your successful training strategy."""
    if epoch < 5:
        return 1e-5  # Protective warmup
    elif epoch < 12:
        return 5e-5  # Increased push for accuracy
    else:
        return 1e-5  # Final stable cooling

# --- 6. TRAINING ---
model.compile(optimizer=tf.keras.optimizers.Adam(), # LR set by scheduler
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule)
]

print(f"\n🚀 Training with 1e-5 -> 5e-5 schedule...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- 7. EXPORT ---
model.save('forager_native_legacy.h5')
print("\n✅ Native Legacy model saved: forager_native_legacy.h5")
