from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel

import torch
import os

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

    # Create dummy input for ONNX export - SIMPLIFIED
    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    print(f"Model input shape: {dummy_input.shape}")
    
    # Test forward pass first
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
    
    # Export to ONNX - SIMPLIFIED VERSION
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,  # Lower opset version for compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )
        print(f"Detection model exported to {onnx_model_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Trying with even simpler settings...")
        
        # Try with minimal settings
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=9,  # Even lower version
            input_names=['input'],
            output_names=['output']
        )
        print(f"Detection model exported to {onnx_model_path} (simplified)")

signal_length = 320

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose model type: "Simple" or "Complex"
model_type = "Complex"

if model_type == "Simple":
    model = SimpleDetectionModel(signal_length=signal_length).to(device)
    modelname = "SimpleDetectionModel"
elif model_type == "Complex":
    model = ComplexDetectionModel(signal_length=320).to(device)
    modelname = "ComplexDetectionModel"

attempt = "001"

# SPECIFY YOUR CHECKPOINT PATH HERE
model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'  # CHANGE THIS PATH

export_detection_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length)
