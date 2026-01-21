import tensorflow as tf

# Load the float model
model = tf.keras.models.load_model('forager_mobilenetv3_float.keras')

# Re-save in legacy HDF5 format (.h5) — compatible with older Keras/TF
model.save('forager_mobilenetv3_legacy.h5')

print("Legacy .h5 model saved — ready for Edge-MDT with TF 2.15!")
