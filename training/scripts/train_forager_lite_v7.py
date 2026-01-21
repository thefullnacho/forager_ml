import os

# 1. FORCE CPU MODE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint

print(f"✅ GPU Disabled. Training on CPU with TensorFlow {tf.__version__}")

# --- CONFIG ---
BASE = Path('inat_dataset')
CLASSES = ['deadly', 'edible', 'medicinal', 'plants']
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_WARMUP = 10
EPOCHS_FINETUNE = 20

# --- CLASS WEIGHTS (bump deadly to 2.0 for more bias) ---
class_weights = {0: 2.0, 1: 1.0, 2: 1.0, 3: 1.0}

# --- DATA LOADING (unchanged) ---
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
    
    deadly_idx = [i for i, l in enumerate(labels) if l == 0]
    paths += [paths[i] for i in deadly_idx * 4]
    labels += [0] * (len(deadly_idx) * 4)
    
    return paths, labels

paths, labels = load_data()

paths_train, paths_val, labels_train, labels_val = train_test_split(
    paths, labels, test_size=0.1, stratify=labels, random_state=42
)

# --- AUGMENTATION & DATASET (sparse labels) ---
aug = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Blur(blur_limit=3, p=0.2),
])

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)  # Safety clip
    
    def augment_np(x):
        augmented = aug(image=x.numpy().astype(np.uint8))
        return augmented["image"].astype(np.float32)
    
    img = tf.cond(
        tf.random.uniform([]) < 0.8,
        lambda: tf.py_function(augment_np, [img], tf.float32),
        lambda: img
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label  # Sparse label

train_ds = tf.data.Dataset.from_tensor_slices((paths_train, labels_train)).shuffle(20_000, seed=42).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((paths_val, labels_val)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_train = len(paths_train)
num_val = len(paths_val)
train_steps = num_train // BATCH_SIZE
val_steps = num_val // BATCH_SIZE

print(f"Dataset Ready: {num_train} Training, {num_val} Validation")

# --- MODEL SETUP ---
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False 

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation=None)
])

model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),  # Lower decay
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # No smoothing
              metrics=['accuracy'])

# --- PHASE 1: WARMUP ---
print("\n🔥 PHASE 1: Warming up head...")
model.fit(train_ds.repeat(), validation_data=val_ds.repeat(), 
          epochs=EPOCHS_WARMUP, steps_per_epoch=train_steps, validation_steps=val_steps,
          class_weight=class_weights,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs/warmup')])

# --- PHASE 2: FINE TUNE ---
print("\n🔓 PHASE 2: Unlocking base...")
base.trainable = True
total_layers = len(base.layers)
unfreeze_from = int(total_layers * 0.5)  # Back to 50%
for layer in base.layers[:unfreeze_from]:
    layer.trainable = False

model.compile(optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # No smoothing
              metrics=['accuracy'])

history = model.fit(
    train_ds.repeat(), 
    validation_data=val_ds.repeat(),
    epochs=EPOCHS_FINETUNE,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),  # Added checkpoint
        tf.keras.callbacks.TensorBoard(log_dir='logs/finetune')
    ]
)

# --- EXPORT ---
print("\n💾 Exporting model to Keras format (linear output)...")
model.save('forager_lite.keras')
print("✅ DONE! Apply softmax in runtime.")

# --- EVALUATION (with TTA) ---
print("\n📊 Generating evaluation report with TTA...")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Updated TTA: Predict on dataset multiple times (leverages random augs in map)
def tta_predict(dataset, num_augs=3):
    all_preds = []
    # We need to extract the actual images from the dataset
    # Note: This can be slow on CPU, so we only do it for the val_ds
    for _ in range(num_augs):
        preds = model.predict(dataset)  # preprocess is already in the dataset mapping
        all_preds.append(tf.nn.softmax(preds).numpy())
    return np.mean(all_preds, axis=0)

val_preds_prob = tta_predict(val_ds)
val_preds = np.argmax(val_preds_prob, axis=1)
val_labels = np.concatenate([y for _, y in val_ds], axis=0)  # Sparse

print("\nClassification Report (TTA):")
print(classification_report(val_labels, val_preds, target_names=CLASSES))

cm = confusion_matrix(val_labels, val_preds)
print("\nConfusion Matrix (TTA):")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASSES, yticklabels=CLASSES)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Forager Lite (EfficientNetB0 with TTA)')
plt.tight_layout()
plt.savefig('confusion_matrix_tta.png')
print("\n✅ Matrix saved as 'confusion_matrix_tta.png'")
