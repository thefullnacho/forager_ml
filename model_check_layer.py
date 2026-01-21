import tensorflow as tf
m = tf.keras.models.load_model('forager_final_npu.h5')
for i, layer in enumerate(m.layers):
    print(f"Layer {i}: {layer.name}")
    if hasattr(layer, 'layers'): # Check inside the MobileNet base
        for sub_layer in layer.layers:
            if 'rescaling' in sub_layer.name.lower():
                print("⚠️ DANGER: Rescaling still found inside base!")
