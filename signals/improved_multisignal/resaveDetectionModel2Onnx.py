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

    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['detection_prob'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},
            'detection_prob': {0: 'batch_size', 1: 'num_signals'},
        }
    )

    print(f"Detection model exported to {onnx_model_path}")

signal_length = 320
d_model = 64
num_heads = 8
num_layers = 4
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_type = "Complex"

if model_type == "Simple":
    model = SimpleDetectionModel(
        signal_length=signal_length,
        d_model=128,
        num_heads=16,
        num_layers=8,
        dropout=dropout
    ).to(device)
    modelname = "SimpleDetectionModel"
elif model_type == "Complex":
    model = ComplexDetectionModel(
        signal_length=signal_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    modelname = "ComplexDetectionModel"

attempt = "000"

model_path = f'models/best_complex_detection.pth'
# model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'


export_detection_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length)
