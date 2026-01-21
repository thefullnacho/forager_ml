import tensorflow as tf

# 1. Load your high-accuracy weights
old_model = tf.keras.models.load_model('forager_native_legacy.h5')

# 2. Build the pure NPU architecture
base = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    include_preprocessing=False, # Physically removes the Rescaling layer
    weights=None
)

# 3. Define the top with the 'Identity' name on the Dense layer
# Note: We name the layer itself 'Identity' so the converter can find the output index
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax', name='Identity')
])

# 4. Transfer your 88.88% accuracy weights
print("🧠 Injecting 88.88% accuracy weights into clean architecture...")
model.set_weights(old_model.get_weights())

# 5. Save as H5 (the most stable format for the Sony toolchain)
model.save('forager_final_npu_fixed.h5')
print("\n✅ Success! Clean model saved as 'forager_final_npu_fixed.h5'.")
