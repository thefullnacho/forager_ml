import tensorflow as tf
import numpy as np

# 1. Load the original high-accuracy model
print("Loading original model...")
original_model = tf.keras.models.load_model('forager_native_legacy.h5')

# 2. Define the exact input the Sony NPU expects
# We force the name "input_1" to be explicit
input_layer = tf.keras.Input(shape=(224, 224, 3), name="input_1")

# 3. Re-trace the graph, physically skipping the Rescaling layer
# We look for the first valid layer (usually a Conv2D) after the rescaling
x = input_layer
found_start = False

for layer in original_model.layers:
    # Skip the Rescaling layer
    if 'rescaling' in layer.name.lower():
        print(f"✂️ REMOVING layer: {layer.name}")
        continue
    
    # Skip the original InputLayer (since we made a new one)
    if isinstance(layer, tf.keras.layers.InputLayer):
        continue

    # Connect the graph
    # We call the layer on 'x' to link it to our new input
    x = layer(x)

# 4. Create the final "Clean" model
# We do NOT add a Lambda layer. We rely on the graph structure.
clean_model = tf.keras.Model(inputs=input_layer, outputs=x)

# 5. Save as Legacy H5
# This ensures no Keras 3.0 "zip" weirdness
clean_model.save('forager_clean_fp32.h5', save_format='h5')
print("\n✅ Success! 'forager_clean_fp32.h5' created.")
print(f"Input Name: {clean_model.input.name}")
print(f"Output Name: {clean_model.output.name}")
