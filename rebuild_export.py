import tensorflow as tf
import tf2onnx
import onnx

# 1. Load the SavedModel we just created
model_path = 'forager_final_hailo'
model = tf.saved_model.load(model_path)

# 2. Define the input signature (Crucial for fixing the 1152-channel error)
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input_rgb"),)

# 3. Convert with Optimization
# This does the 'folding' and 'fusion' internally
onnx_model, _ = tf2onnx.convert.from_saved_model(
    model_path,
    input_signature=spec,
    opset=17, # Opset 17 is standard for Hailo 3.33
    optimize=True # This is the Python equivalent of --fold_const
)

# 4. Save the model
onnx.save(onnx_model, "forager_final.onnx")
print("✅ forager_final.onnx created with optimized graph.")
