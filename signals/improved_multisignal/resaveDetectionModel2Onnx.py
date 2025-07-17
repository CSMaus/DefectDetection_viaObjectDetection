from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel

import torch
import torch.nn as nn

def export_detection_model_to_onnx(model, device, model_path, onnx_model_path, signal_length):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        val_accuracy = checkpoint.get('val_accuracy', 'Unknown')
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy: {val_accuracy}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    print(f"Model input shape: {dummy_input.shape}")
    
    # Test forward pass first
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
    
    # Export to ONNX with opset 13 (supports unflatten)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=13,  # Use opset 13 for unflatten support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print(f"Detection model exported to {onnx_model_path}")


signal_length = 320

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose model type: "Simple" or "Complex"
model_type = "Complex"

if model_type == "Simple":
    model = SimpleDetectionModel(signal_length=signal_length).to(device)
    modelname = "SimpleDetectionModel"
elif model_type == "Complex":
    # Use original model - opset 13 should handle it
    model = ComplexDetectionModel(signal_length=320).to(device)
    modelname = "ComplexDetectionModel"

attempt = "001"

model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'

export_detection_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length)
