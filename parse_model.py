from hailo_sdk_client import ClientRunner

# 1. Setup paths
model_name = "forager_lite"
saved_model_path = "forager_for_hailo"

# 2. Initialize the runner
runner = ClientRunner(hw_arch='hailo8l')

# 3. Parse using 'tensor_shapes'
# This maps the input name to the 224x224x3 shape
runner.translate_tf_model(
    model_path=saved_model_path,
    net_name=model_name,
    tensor_shapes={'input_rgb': [1, 224, 224, 3]}
)

# 4. Save the HAR
runner.save_har('forager_lite.har')
print("✅ Success! forager_lite.har created with 3-channel input.")
