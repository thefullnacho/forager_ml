import h5py

model_path = 'forager_quantized.keras' # or .h5 if you saved as that

with h5py.File(model_path, 'r') as f:
    # Look for the input names
    if 'model_config' in f.attrs:
        print("Model configuration found. Extracting names...")
        # This is a bit of a 'hack' to find the string names in the raw metadata
        import json
        config = json.loads(f.attrs['model_config'])
        layers = config['config']['layers']
        print(f"Input Layer Name: {layers[0]['name']}")
        print(f"Output Layer Name: {layers[-1]['name']}")
    else:
        print("Directly listing groups in the file:")
        for key in f.keys():
            print(f"Group found: {key}")
