import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = '/home/alex/Documents/Forager/forager_dataset/inat_dataset' 

# 1. Load Dataset (Capture class names before mapping)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True, label_mode='categorical'
)
class_names = raw_val_ds.class_names
val_ds = raw_val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# 2. Load the Final Candidate
print("Loading 'forager_final_candidate.keras'...")
model = tf.keras.models.load_model('forager_final_candidate.keras')

# 3. Collect Predictions
true_labels, pred_labels = [], []
for imgs, labs in val_ds:
    preds = model.predict(imgs, verbose=0)
    pred_labels.extend(np.argmax(preds, axis=1))
    true_labels.extend(np.argmax(labs.numpy(), axis=1))

# 4. Generate the Matrix
cm = confusion_matrix(true_labels, pred_labels)

# 5. Calculate Safety Metrics
# Deadly is index 0, Edible is index 1
deadly_as_edible = cm[0][1]
deadly_recall = cm[0][0] / np.sum(cm[0])

print("\n" + "="*30)
print(f"🚨 SAFETY CHECK 🚨")
print(f"Fatal Errors (Deadly as Edible): {deadly_as_edible}")
print(f"Deadly Detection Rate (Recall): {deadly_recall:.1%}")
print("="*30)

# 6. Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='YlGnBu')
plt.title('Forager Final Candidate - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n--- FINAL CLASSIFICATION REPORT ---")
print(classification_report(true_labels, pred_labels, target_names=class_names))
