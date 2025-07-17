from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel

import torch
import os

def export_model_to_onnx(model, device, model_path, onnx_model_path, signal_length):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    # Export to ONNX
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

    print(f"Model exported to {onnx_model_path}")

signal_length = 320

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ComplexDetectionModel(signal_length=signal_length).to(device)

modelname = "ComplexDetectionModel"
attempt = "001"
model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'
export_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length)
