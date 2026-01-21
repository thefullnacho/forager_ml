import tensorflow as tf

# 1. Load your 88.88% record-breaker (H5 is standard and won't crash)
model = tf.keras.models.load_model('forager_native_legacy.h5')

# 2. Build a new Functional model skipping the Rescaling layer
# We define a new input that matches the layer AFTER Rescaling
# In MobileNetV3, the layer after Rescaling is usually the first Conv layer
new_input = tf.keras.Input(shape=(224, 224, 3), name="input_1")

# We iterate through the layers, skipping 'rescaling'
x = new_input
for layer in model.layers:
    if 'rescaling' in layer.name.lower():
        print(f"✂️ Stripping layer: {layer.name}")
        continue
    x = layer(x)

# 3. Save the "NPU-Ready" High Accuracy Model
clean_model = tf.keras.Model(inputs=new_input, outputs=x)
clean_model.save('forager_clean_88.h5')
print("\n✅ Success! 'forager_clean_88.h5' is ready for quantization.")
