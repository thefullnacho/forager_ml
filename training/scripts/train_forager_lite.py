import os

# 1. FORCE CPU MODE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import EfficientNetB0  # No preprocess_input needed

print(f"✅ GPU Disabled. Training on CPU with TensorFlow {tf.__version__}")

# --- CONFIG ---
BASE = Path('inat_dataset')
CLASSES = ['deadly', 'edible', 'medicinal', 'plants']
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_WARMUP = 10
EPOCHS_FINETUNE = 20

# --- WEIGHTED FOCAL LOSS (Class-specific alpha) ---
def weighted_focal_loss(gamma=2.0, alpha=[0.5, 0.1, 0.2, 0.2]):
    # alpha ordered: ['deadly', 'edible', 'medicinal', 'plants']
    # Deadly gets 5x the weight of edible — aggressively penalizes missing deadly
    alpha = tf.constant(alpha, dtype=tf.float32)
    
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true)  # Safety: remove any extra dims
        y_true = tf.one_hot(y_true, depth=4)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Focal weighting
        focal_weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = -y_true * focal_weight * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return focal_loss_fn

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

# Stratified split
paths_train, paths_val, labels_train, labels_val = train_test_split(
    paths, labels, test_size=0.1, stratify=labels, random_state=42
)

# --- 3. AUGMENTATION & DATASET ---
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
    img = tf.cast(img, tf.float32)  # Keep [0, 255] — EfficientNetB0 handles internal rescaling
    
    def augment_np(x):
        augmented = aug(image=x.numpy().astype(np.uint8))
        return augmented["image"].astype(np.float32)
    
    img = tf.cond(
        tf.random.uniform([]) < 0.8,
        lambda: tf.py_function(augment_np, [img], tf.float32),
        lambda: img
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((paths_train, labels_train)).shuffle(20_000, seed=42).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((paths_val, labels_val)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_train = len(paths_train)
num_val = len(paths_val)
train_steps = num_train // BATCH_SIZE
val_steps = num_val // BATCH_SIZE

print(f"Dataset Ready: {num_train} Training, {num_val} Validation")

# --- 4. MODEL SETUP ---
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False 

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation=None)  # Linear (no softmax) — better for Hailo quantization
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=weighted_focal_loss(gamma=2.0, alpha=[0.5, 0.1, 0.2, 0.2]),
              metrics=['accuracy'])

# --- 5. PHASE 1: WARMUP ---
print("\n🔥 PHASE 1: Warming up head...")
model.fit(train_ds.repeat(), validation_data=val_ds.repeat(), 
          epochs=EPOCHS_WARMUP, steps_per_epoch=train_steps, validation_steps=val_steps,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs/warmup')])

# --- 6. PHASE 2: FINE TUNE ---
print("\n🔓 PHASE 2: Unlocking base...")
base.trainable = True
total_layers = len(base.layers)
unfreeze_from = int(total_layers * 0.5)  # Top 50%
for layer in base.layers[:unfreeze_from]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=weighted_focal_loss(gamma=2.0, alpha=[0.5, 0.1, 0.2, 0.2]),
              metrics=['accuracy'])

history = model.fit(
    train_ds.repeat(), 
    validation_data=val_ds.repeat(),
    epochs=EPOCHS_FINETUNE,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='logs/finetune')
    ]
)

# --- 7. EXPORT TO KERAS (HAILO-READY) ---
print("\n💾 Exporting model to Keras format (linear output)...")
model.save('forager_lite.keras')
print("✅ DONE! 'forager_lite.keras' ready. Apply softmax in runtime inference for best Hailo quantization.")

# --- 8. POST-TRAINING EVALUATION (CONFUSION MATRIX + REPORT) ---
print("\n📊 Generating evaluation report...")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions (apply softmax manually)
val_preds_logits = model.predict(val_ds)
val_preds_prob = tf.nn.softmax(val_preds_logits).numpy()
val_preds = np.argmax(val_preds_prob, axis=1)

val_labels = np.concatenate([y for _, y in val_ds], axis=0)

print("\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=CLASSES))

cm = confusion_matrix(val_labels, val_preds)
print("\nConfusion Matrix:")
print(cm)

# Plot and save
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=CLASSES, yticklabels=CLASSES)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Forager Lite (EfficientNetB0)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n✅ Confusion matrix plotted and saved as 'confusion_matrix.png'")
