import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import os

# 1. Setup
os.environ["TF_USE_LEGACY_KERAS"] = "1"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 2. Data Loader
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

# NORMALIZE (0-1 Range)
def process_data(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(process_data)
val_ds = val_ds.map(process_data)

# 3. Build Model (Functional API)
print("🏗️  Building Production Architecture...")
inputs = Input(shape=(224, 224, 3), name="input_1")

# Base: MobileNetV3 Large (Clean, no internal preprocessing)
base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs,
    include_top=False,
    weights='imagenet',
    include_preprocessing=False
)

# Unfreeze the top layers for better fine-tuning (Optional, but helps accuracy)
# We freeze the bottom 90% and train the top 10%
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 4. Compile
# We use a lower learning rate because we unfroze some layers
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Callbacks (The Safety Net)
checkpoint = callbacks.ModelCheckpoint(
    'forager_production.keras',     # Save to this specific filename
    monitor='val_accuracy',         # Watch validation accuracy
    save_best_only=True,            # Only save if it's better than before
    mode='max',                     # Higher is better
    verbose=1
)

# 6. Train Long
print("🚀 Starting Production Training (20 Epochs)...")
model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=20, 
    callbacks=[checkpoint] # Add the checkpoint
)

print("\n✅ Success! The BEST model was saved as 'forager_production.keras'.")
