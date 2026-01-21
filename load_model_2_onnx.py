import tensorflow as tf

# Load the .keras model
model = tf.keras.models.load_model('forager_lite.keras')

# Save as SavedModel
tf.saved_model.save(model, 'forager_lite_saved_model')
print("✅ SavedModel exported to 'forager_lite_saved_model'")
