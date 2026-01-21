import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 1. Load the raw dataset first
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=123,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    label_mode='categorical'
)

# 2. CAPTURE CLASS NAMES NOW (Before mapping)
class_names = raw_val_ds.class_names #
print(f"Verified Classes: {class_names}")

# 3. Apply normalization
val_ds = raw_val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# 4. Load Master Model
print("Loading 'forager_master_v1.keras'...")
model = tf.keras.models.load_model('forager_master_v1.keras')

# 5. Collect predictions
true_labels, pred_labels = [], []
for imgs, labs in val_ds:
    # Use verbose=0 to keep the terminal clean
    preds = model.predict(imgs, verbose=0)
    pred_labels.extend(np.argmax(preds, axis=1))
    true_labels.extend(np.argmax(labs.numpy(), axis=1))

# 6. Visualization
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.title('Forager Master Run Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("\n--- MASTER CLASSIFICATION REPORT ---")
print(classification_report(true_labels, pred_labels, target_names=class_names))
