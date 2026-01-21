import os
import cv2
import numpy as np
from tqdm import tqdm

calib_dir = 'calib_images'
output_npy = 'calib_set.npy'
target_size = (224, 224)  # Matches your model input

image_files = [f for f in os.listdir(calib_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()  # Consistent order

images = []

print(f"Processing {len(image_files)} calibration images...")
for f in tqdm(image_files):
    path = os.path.join(calib_dir, f)
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to read {path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1] — matches your training preprocess
    images.append(img)

images = np.array(images)
print(f"Saving calibration set: {images.shape} -> {output_npy}")
np.save(output_npy, images)
print("Done! calib_set.npy ready for Hailo.")
