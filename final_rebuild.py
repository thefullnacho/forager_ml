import tensorflow as tf

# Define Architecture
base = tf.keras.applications.MobileNetV3Large(
    weights=None, 
    include_top=False,
    input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Inject the weights
model.load_weights('forager_weights.weights.h5')

# IMPORTANT: We save as .keras, but since we are in TF 2.15, 
# it will naturally save in the format the Sony tool expects.
model.save('forager_for_sony.keras')
print("Model saved as forager_for_sony.keras")
