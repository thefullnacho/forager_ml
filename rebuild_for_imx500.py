import tensorflow as tf
from pathlib import Path

print(f"TF version: {tf.__version__}")  # Should be ~2.15

# Re-create the exact architecture
base = tf.keras.applications.MobileNetV3Large(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False  # Match training setup (though weights include finetuned)

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Load weights from the new .keras (it extracts weights compatibly)
original_model = tf.keras.models.load_model('forager_mobilenetv3_float.keras')
model.set_weights(original_model.get_weights())

# Save in format compatible with older TF
model.save('forager_mobilenetv3_imx500_compat.keras')

print("Compatible model saved for IMX500 conversion!")
