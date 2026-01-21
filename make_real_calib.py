import tensorflow as tf
import numpy as np
import os
import glob

# 1. Update this to your actual image folder path
# We need to find about 64 real JPGs.
img_dir = "/home/alex/Documents/Forager/forager_dataset/calib_images" 
# This search pattern looks for images in subfolders (e.g., Deadly/Edible)
search_pattern = os.path.join(img_dir, "*", "*.jpg")

image_paths = glob.glob(search_pattern)

if len(image_paths) == 0:
    # Fallback: Try looking in the root directory if no subfolders exist
    search_pattern = os.path.join(img_dir, "*.jpg")
    image_paths = glob.glob(search_pattern)

if len(image_paths) == 0:
    print(f"❌ Error: No images found in {img_dir}. Please check the path.")
    exit()

print(f"✅ Found {len(image_paths)} images. Processing the first 64...")

# 2. Load, Resize, and Pack
images_list = []
target_count = 64
count = 0

for p in image_paths:
    if count >= target_count:
        break
    try:
        # Load raw bytes
        img_raw = tf.io.read_file(p)
        # Decode image (uint8 0-255)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        # Resize to 224x224 (The model's exact input size)
        img = tf.image.resize(img, (224, 224))
        
        images_list.append(img)
        count += 1
    except Exception as e:
        print(f"Skipping bad file {p}: {e}")

# 3. Save as .npy
if images_list:
    # Convert to float32 (0-255 range) which is standard for calibration
    calib_data = np.array(images_list).astype(np.float32)
    
    output_path = '/home/alex/Documents/Forager/forager_dataset/calib_data_real.npy'
    np.save(output_path, calib_data)
    print(f"🚀 Success! Saved {output_path} with shape {calib_data.shape}")
else:
    print("❌ Failed to process any images.")
