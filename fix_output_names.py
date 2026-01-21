import tensorflow as tf

# 1. Load the model you just rebuilt (the one that's clean)
model = tf.keras.models.load_model('forager_final_npu.h5')

# 2. Rename the output layer to 'Identity'
# This ensures the Sony DspAdvisor can "find" the output index
new_output = tf.keras.layers.Lambda(lambda x: x, name='Identity')(model.output)
final_model = tf.keras.Model(inputs=model.input, outputs=new_output)

# 3. Save the final version
final_model.save('forager_final_fixed.h5')
print("\n✅ Success! Output renamed to 'Identity'. Ready for final run.")
