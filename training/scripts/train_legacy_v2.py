import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os

# 1. Setup
os.environ["TF_USE_LEGACY_KERAS"] = "1"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
# Update this to your correct path
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

# --- CRITICAL FIX: Normalize Data Outside the Model ---
# This divides pixels by 255.0 so they are 0-1, matching MobileNet expectations.
def process_data(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = train_ds.map(process_data)
val_ds = val_ds.map(process_data)
# ------------------------------------------------------

# 3. Build Model (Functional API)
print("🏗️  Building architecture...")
inputs = Input(shape=(224, 224, 3), name="input_1")

# We feed the ALREADY NORMALIZED data into the base
base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs,
    include_top=False,
    weights='imagenet',
    include_preprocessing=False # Still False, to keep Sony happy!
)

base_model.trainable = False 

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 4. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train
print("🚀 Training (Corrected)...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 6. Save
print("💾 Saving...")
model.save('forager_native_v2.keras')
print("✅ Success! 'forager_native_v2.keras' created.")
