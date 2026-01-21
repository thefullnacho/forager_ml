import tensorflow as tf
import model_compression_toolkit as mct
import numpy as np
import os

# 1. Load the Golden Model
print("🧠 Loading 'forager_native_v2.keras'...")
model = tf.keras.models.load_model('forager_final_candidate.keras')

# 2. Setup Calibration
CALIB_DIR = '/home/alex/Documents/Forager/forager_dataset/calib_images'
img_paths = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def representative_data_gen():
    for path in img_paths[:50]:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        # NORMALIZE: Matches the training logic (0-1 range)
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

# 4. EXPORT as .keras
# Since the input was Legacy Keras, this export will maintain that structure
print("📦 Exporting...")
mct.exporter.keras_export_model(
    model=quantized_model, 
    save_model_path='forager_quantized.keras'
)
print("\n✅ Success! 'forager_quantized.keras' is ready.")
