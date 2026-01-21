import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")  # Should be 2.15.x

# Re-create architecture EXACTLY as in training
base = tf.keras.applications.MobileNetV3Large(
    weights='imagenet',  # Loads pretrained (matches your training preload)
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False  # Matches warmup phase, but weights include finetune

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Load weights from the Keras 3-saved model (this bypasses config)
float_model = tf.keras.models.load_model('forager_mobilenetv3_float.keras')
model.set_weights(float_model.get_weights())

# Save in legacy-compatible format
model.save('forager_imx500_compat.h5', save_format='h5')  # Explicit H5

print("Legacy-compatible H5 model saved!")
