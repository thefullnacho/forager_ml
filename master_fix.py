import tensorflow as tf
import model_compression_toolkit as mct
import numpy as np
import os
import shutil

# 1. Load the original weights (The 88% ones)
print("🧠 Loading original weights...")
original_model = tf.keras.models.load_model('forager_native_legacy.h5')

# 2. Rebuild the Architecture Purely (No Rescaling Layer)
print("🏗️  Rebuilding Pure Architecture...")
# include_preprocessing=False physically prevents the layer from existing
base = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    include_preprocessing=False, 
    weights=None
)

# Reconstruct the Head
clean_model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 3. Transfer Weights
# We perform a dummy forward pass to initialize shapes
clean_model(tf.zeros((1, 224, 224, 3)))
print("💉 Injecting weights...")
# This might warn about shape mismatches if the original had extra layers, 
# but usually works for standard transfer learning setups.
# If this fails, we will use the layer-by-layer transfer.
clean_model.set_weights(original_model.get_weights())

# 4. Quantize
print("📉 Running Quantization...")
CALIB_DIR = '/home/alex/Documents/Forager/forager_dataset/calib_images'
img_paths = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def representative_data_gen():
    for path in img_paths[:50]: # 50 is enough for this final fix
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]

tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500', 'v1')
quantized_model, _ = mct.ptq.keras_post_training_quantization(
    clean_model, 
    representative_data_gen,
    target_platform_capabilities=tpc
)

# 5. Export and Trojan Rename
print("📦 Exporting Trojan File...")
# Save as H5 (Legacy format)
mct.exporter.keras_export_model(
    model=quantized_model, 
    save_model_path='forager_final_trojan.h5'
)

# Rename to .keras to trick the converter
if os.path.exists('forager_final_trojan.keras'):
    os.remove('forager_final_trojan.keras')
    
os.rename('forager_final_trojan.h5', 'forager_final_trojan.keras')
print("\n✅ Success! 'forager_final_trojan.keras' is ready.")
print("   (It is secretly an H5 file with no Rescaling layer)")
