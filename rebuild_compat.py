import tensorflow as tf

print(f"TF version: {tf.__version__}")  # Should be 2.15.x

# Re-create architecture exactly
base = tf.keras.applications.MobileNetV3Large(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Load the float model from training env
float_model = tf.keras.models.load_model('forager_mobilenetv3_float.keras')

# Copy weights (this works across versions)
model.set_weights(float_model.get_weights())

# Save compatible format
model.save('forager_imx500_compat.keras')

print("Compatible model saved!")
