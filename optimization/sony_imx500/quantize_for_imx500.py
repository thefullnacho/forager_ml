import tensorflow as tf

# Load the float model (should work now)
model = tf.keras.models.load_model('forager_mobilenetv3_float.keras')

# Simple save as quantized-ready (MCT handles quantization via converter flags)
model.save('forager_mobilenetv3_imx500_ready.keras')

print("Model ready for conversion!")
