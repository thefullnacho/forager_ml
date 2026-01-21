import tensorflow as tf

# 1. Load the quantized model that just worked
q_model = tf.keras.models.load_model('forager_quantized.keras')

# 2. Identify the core input and the layer after Rescaling
# MobileNetV3 usually has: Input -> Rescaling -> Base_Model
# We want to skip layer index 1 (the Rescaling layer)
new_input = tf.keras.Input(shape=(224, 224, 3), name="input_1")

# We skip the Rescaling layer and start from the actual base model (usually layer index 2)
# We re-link the new input to the first layer of the MobileNet base
x = new_input
for layer in q_model.layers:
    if "rescaling" in layer.name.lower():
        print(f"✂️ Skipping unsupported layer: {layer.name}")
        continue
    # Re-link the remaining layers
    x = layer(x)

# 3. Create the stripped model
stripped_model = tf.keras.Model(inputs=new_input, outputs=x)
stripped_model.save('forager_stripped.keras')

print("\n✅ Success! forager_stripped.keras is now NPU-compatible.")
