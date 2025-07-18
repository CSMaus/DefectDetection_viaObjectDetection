from detection_models.hybrid_binary import HybridBinaryModel

import torch
import os

def export_model_to_onnx(model, device, model_path, onnx_model_path, signal_length, hidden_sizes, num_heads=8):
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
        opset_version=15,
        input_names=['input'],
        output_names=['defect_prob'],  # Only one output instead of 3
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},
            'defect_prob': {0: 'batch_size', 1: 'num_signals'},
        }
    )

    print(f"Model exported to {onnx_model_path}")

signal_length = 320
hidden_sizes = [128, 64, 32]
num_heads = 8
dropout = 0.1
num_transformer_layers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HybridBinaryModel(
    signal_length=signal_length,
    hidden_sizes=hidden_sizes,
    num_heads=num_heads,
    dropout=dropout,
    num_transformer_layers=num_transformer_layers
).to(device)

modelname = "HybridBinaryModel"
attempt = "002"
model_path = f'models/HybridBinaryModel_20250718_2100/best_detection.pth'  # UPDATE THIS PATH
export_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length, hidden_sizes, num_heads)
