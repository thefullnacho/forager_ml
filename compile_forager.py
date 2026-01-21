import tensorflow as tf
from hailo_sdk_client import ClientRunner

model_name = "forager_lite"
tflite_path = "forager_lite.tflite"
chosen_hw_arch = "hailo8l"  # Crucial for Pi 5

print(f"🚀 Starting compilation for {chosen_hw_arch}...")

runner = ClientRunner(hw_arch=chosen_hw_arch)
# Parse the TFLite model
har_path = runner.translate_tf_model(tflite_path, model_name)
print(f"✅ Parsed to HAR: {har_path}")

# Optimize (loads the model as-is since it's already int8)
runner.optimize_full_precision() 

# Compile to HEF
hef = runner.compile()

# Save
hef_path = f"{model_name}.hef"
with open(hef_path, "wb") as f:
    f.write(hef)

print(f"🏆 SUCCESS! Saved to: {hef_path}")
