import tensorflow as tf

# 1. Point to your clean SavedModel folder
saved_model_dir = "forager_final_hailo"

# 2. Convert to TFLite (Float32 ONLY)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# CRITICAL: Do NOT set converter.optimizations
# We want the weights to remain raw float32 numbers
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS 
]

tflite_model = converter.convert()

# 3. Save as a distinct filename so we know it's the float version
with open("forager_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Success: forager_float32.tflite created (Pure Float32).")
