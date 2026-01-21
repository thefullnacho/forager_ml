import zipfile
import json

# Open the .keras file as a zip archive
with zipfile.ZipFile('forager_quantized.keras', 'r') as z:
    # Read the config.json inside the archive
    config_data = z.read('config.json')
    config = json.loads(config_data)
    
    # Navigate to the layer list
    layers = config['config']['layers']
    
    # Print the First (Input) and Last (Output) layer names
    print(f"✅ FOUND Input Name:  {layers[0]['name']}")
    print(f"✅ FOUND Output Name: {layers[-1]['name']}")
