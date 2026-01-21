import tensorflow as tf
import model_compression_toolkit as mct
import numpy as np
import os
import shutil

# 1. Load the clean architecture (No Rescaling)
print("🧠 Loading clean architecture...")
model = tf.keras.models.load_model('forager_clean_fp32.h5')

# 2. Setup Calibration (Standard 50 images)
print("⚙️  Calibration setup...")
CALIB_DIR = '/home/alex/Documents/Forager/forager_dataset/calib_images'
img_paths = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def representative_data_gen():
    for path in img_paths[:50]:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]

# 3. Quantize
print("📉 Running Quantization...")
tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500', 'v1')
quantized_model, _ = mct.ptq.keras_post_training_quantization(
    model, 
    representative_data_gen,
    target_platform_capabilities=tpc
)

# 4. EXPORT AS SAVED_MODEL
# We create a specific directory for the model
output_dir = 'forager_saved_model_v1'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print(f"📦 Exporting to SavedModel directory: {output_dir}...")

# We use the underlying TF save, not the Keras save. 
# This bypasses the version mismatch.
tf.saved_model.save(quantized_model, output_dir)

print("\n✅ Success! Directory 'forager_saved_model_v1' created.")
