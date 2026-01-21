import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os

# 1. Setup - Explicitly force Keras 2 behavior just in case
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
    label_mode='categorical' # Important for Softmax
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

# 3. Build Model (Functional API - The safest for hardware)
print("🏗️  Building architecture...")
inputs = Input(shape=(224, 224, 3), name="input_1")

# Preprocessing: We do it via a Lambda layer if needed, or rely on external norm.
# For IMX500, we prefer EXTERNAL norm (in your Pi script).
# So we feed raw inputs into MobileNet.

base_model = tf.keras.applications.MobileNetV3Large(
    input_tensor=inputs,
    include_top=False,
    weights='imagenet',
    include_preprocessing=False # Crucial: No Rescaling layer!
)

base_model.trainable = False # Freeze base

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax', name="Identity")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# 4. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train (Fast Fine-Tune)
print("🚀 Training...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 6. Save as Native Keras
print("💾 Saving...")
# In TF 2.15, .keras is the default format but uses the older, compatible structure.
model.save('forager_native_v2.keras') 
print("✅ Success! 'forager_native_v2.keras' created in Legacy Mode.")
