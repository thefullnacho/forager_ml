import tensorflow as tf

# Load legacy .h5 model
model = tf.keras.models.load_model('forager_mobilenetv3_legacy.h5')

# Save as .keras for converter (Edge-MDT expects .keras)
model.save('forager_mobilenetv3_imx500_ready.keras')

print("Model ready for IMX500 conversion!")
