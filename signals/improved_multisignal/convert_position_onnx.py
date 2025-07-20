from detection_models.position_localization import PositionLocalizationModel

import torch
import os

def export_model_to_onnx(model, device, model_path, onnx_model_path, signal_length, d_model=128, num_heads=8):
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
        output_names=['defect_start', 'defect_end'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},
            'defect_start': {0: 'batch_size', 1: 'num_signals'},
            'defect_end': {0: 'batch_size', 1: 'num_signals'},
        }
    )

    print(f"Model exported to {onnx_model_path}")

signal_length = 320
d_model = 128
num_heads = 8
num_layers = 4
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model = PositionLocalizationModel(
#         signal_length=signal_length,
#         d_model=d_model,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         dropout=dropout
#     ).to(device)

model = PositionLocalizationModel(
        signal_length=320,
        hidden_sizes=[128, 64, 32],
        num_heads=8,
        dropout=0.1,
        num_transformer_layers=4
    ).to(device)

modelname = "PositionLocalizationModel"
attempt = "002"
model_path = f'models/PositionLocalization_20250720_1404/best_position_model.pth'
export_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length, d_model, num_heads)
