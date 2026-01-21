import tensorflow as tf
import os

# 1. Load the model you already trained
old_model = tf.keras.models.load_model('forager_lite.keras', compile=False)

# 2. Extract the base and reconstruct to ensure 3-channel input
# In most EfficientNet loads, the 'base' is the first layer
base_layer = old_model.layers[0]

inputs = tf.keras.Input(shape=(224, 224, 3), name="input_rgb")
# Direct pass to ensure the stem isn't skipped
x = base_layer(inputs)

# Re-apply the head layers (GAP, Dropout, Dense)
for layer in old_model.layers[1:]:
    x = layer(x)

full_model = tf.keras.Model(inputs=inputs, outputs=x)

# 3. USE THE EXPORT API (Keras 3 way to get a SavedModel folder)
export_path = 'forager_for_hailo'
full_model.export(export_path)

print(f"✅ Model exported to {export_path}/ folder for Hailo parsing.")
