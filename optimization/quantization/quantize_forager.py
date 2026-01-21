import tensorflow as tf
import model_compression_toolkit as mct
import numpy as np
import os

# 1. Load the clean model (No Rescaling, Named 'Identity')
model = tf.keras.models.load_model('forager_final_npu_fixed.h5')

# 2. Standard Representative Data Gen (100 images is plenty for final check)
CALIB_DIR = '/home/alex/Documents/Forager/forager_dataset/calib_images'
img_paths = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

def representative_data_gen():
    for path in img_paths[:100]:
        img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path)), [224, 224])
        yield [np.expand_dims(tf.cast(img, tf.float32) / 255.0, axis=0)]

# 3. Quantize
tpc = mct.get_target_platform_capabilities('tensorflow', 'imx500', 'v1')
quantized_model, _ = mct.ptq.keras_post_training_quantization(
    model, representative_data_gen, target_platform_capabilities=tpc
)

# 4. THE FIX: Save as LEGACY H5
# This avoids the 'str' object error and the 'file signature' error
quantized_model.save('forager_quantized_legacy.h5', save_format='h5')
print("✅ Legacy H5 Quantized Model Saved.")
