import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Keep commented or remove— we'll force it cleaner

import tensorflow as tf

# Nuclear force-hide GPU (works even if env var leaks)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices([], 'GPU')
    print("✅ GPU forcibly hidden — running on pure CPU")

import numpy as np
import albumentations as A
from pathlib import Path
import shutil

print(f"✅ Training on CPU with TensorFlow {tf.__version__}")

# --- CONFIG ---
BASE = Path('/home/alex/Documents/Forager/forager_dataset/inat_dataset')  # Your dataset path
CLASSES = ['deadly', 'edible', 'medicinal', 'plants']
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch size is often faster on CPU
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 15

# --- 2. DATA LOADING ---
def load_data():
    paths = []
    labels = []
    print('Scanning folders...')
    for idx, cls in enumerate(CLASSES):
        folder = BASE / cls
        count = 0
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(entry.path)
                    labels.append(idx)
                    count += 1
        print(f'  {cls}: {count} images')
    
    # Oversample deadly 5x
    deadly_idx = [i for i, l in enumerate(labels) if l == 0]
    paths += [paths[i] for i in deadly_idx * 4]
    labels += [0] * (len(deadly_idx) * 4)
    
    return paths, labels

paths, labels = load_data()

# --- 3. AUGMENTATION & DATASET ---
# Fixed for Albumentations 1.4+ (Strict Pydantic)
aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
])

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    
    def augment_np(x):
        augmented = aug(image=(x.numpy() * 255).astype(np.uint8))
        return augmented["image"].astype(np.float32) / 255.0
    
    # 80% chance of augmentation
    img = tf.cond(
        tf.random.uniform([]) < 0.8,
        lambda: tf.py_function(augment_np, [img], tf.float32),
        lambda: img
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label

# Build Pipeline
dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.shuffle(20_000, seed=42)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

val_split = int(0.1 * len(paths))
train_ds = dataset.skip(val_split).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.take(val_split).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Steps calculation
num_train = len(paths) - val_split
train_steps = num_train // BATCH_SIZE
val_steps = val_split // BATCH_SIZE

print(f"Dataset Ready: {num_train} Training, {val_split} Validation")

# --- 4. MODEL SETUP (2-STAGE) ---
# Load Base (Frozen)
base = tf.keras.applications.MobileNetV3Large(
    weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False 

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. PHASE 1: WARMUP ---
print("\n🔥 PHASE 1: Warming up head...")
model.fit(train_ds.repeat(), validation_data=val_ds.repeat(), 
          epochs=EPOCHS_WARMUP, steps_per_epoch=train_steps, validation_steps=val_steps)

# --- 6. PHASE 2: FINE TUNE ---
print("\n🔓 PHASE 2: Unlocking base...")
base.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Low LR
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds.repeat(), 
    validation_data=val_ds.repeat(),
    epochs=EPOCHS_FINETUNE,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1)
    ]
)

print("\n💾 Saving full floating-point Keras model for Hailo...")
model.save('forager_mobilenetv3_float.keras')  # Modern Keras format
model.save_weights('forager_mobilenetv3_weights.h5')  # Backup weights

print("✅ Float model saved — ready for Hailo ONNX export & DFC!")

# --- 7. EXPORT TO ONNX for Hailo ---
print("\n💾 Exporting to ONNX for Hailo...")
tf.saved_model.save(model, 'forager_mobilenetv3_saved_model')
os.system('python -m tf2onnx.convert --saved-model forager_mobilenetv3_saved_model --output forager_mobilenetv3.onnx --opset 13')

print("✅ DONE! 'forager_mobilenetv3.onnx' is ready for Hailo DFC compilation.")
