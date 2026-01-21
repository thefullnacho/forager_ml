import tensorflow as tf
import numpy as np
from pathlib import Path

# Config
CALIB_DIR = Path('~/Documents/Forager/forager_dataset/calib_images').expanduser()
IMG_SIZE = 224
MAX_IMAGES = 2000  # Use all or cap as needed

# Load and preprocess images
calib_data = []
count = 0
for file in CALIB_DIR.glob('*'):
    if file.suffix.lower() in ('.jpg', '.jpeg', '.png') and count < MAX_IMAGES:
        img = tf.io.read_file(str(file))
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        img = tf.clip_by_value(img, 0.0, 255.0)
        calib_data.append(img.numpy())
        count += 1
        print(f'Processed {count}/{MAX_IMAGES}: {file.name}')

calib_data = np.stack(calib_data, axis=0)
np.save('calib_data.npy', calib_data)
print(f"✅ calib_data.npy saved with shape: {calib_data.shape}")
