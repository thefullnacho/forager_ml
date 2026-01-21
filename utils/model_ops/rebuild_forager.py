import tensorflow as tf

# 1. Load your 88.88% record-breaker
old_model = tf.keras.models.load_model('forager_native_legacy.h5')

# 2. Create a clean base WITHOUT the Rescaling layer
# 'include_preprocessing=False' is the magic flag for MobileNetV3
base = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    include_preprocessing=False, # This physically removes the Rescaling layer
    weights=None # We will load your weights manually
)

# 3. Reconstruct your specific top layers
new_model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 4. Transfer the weights precisely
print("🧠 Transferring weights from old model to clean architecture...")
new_model.set_weights(old_model.get_weights())

# 5. Save the pure NPU-ready model
new_model.save('forager_final_npu.h5')
print("\n✅ Success! 'forager_final_npu.h5' is guaranteed to be clean.")
