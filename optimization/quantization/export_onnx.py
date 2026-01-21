import tensorflow as tf
import tf2onnx
import onnx

# 1. Load your best saved model
model = tf.keras.models.load_model('forager_final_candidate.keras')

# 2. Define the input signature for 640x640 resolution
spec = (tf.TensorSpec((None, 640, 640, 3), tf.float32, name="input_1"),)

# 3. Convert to ONNX
output_path = "shared_with_docker/forager_generalist.onnx"
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# 4. Save
onnx.save(onnx_model, output_path)
print(f"✅ Generalist model exported to {output_path}")
