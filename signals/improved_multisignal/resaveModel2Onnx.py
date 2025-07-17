from improved_model import ImprovedMultiSignalClassifier
from enhanced_position_model_acc091 import EnhancedPositionMultiSignalClassifier
from fixed_enhanced_position_model import FixedEnhancedPositionMultiSignalClassifier

import torch
import os

def export_model_to_onnx(model, device, model_path, onnx_model_path, signal_length, hidden_sizes, num_heads=4):
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
        output_names=['defect_prob', 'defect_start', 'defect_end'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_signals'},
            'defect_prob': {0: 'batch_size', 1: 'num_signals'},
            'defect_start': {0: 'batch_size', 1: 'num_signals'},
            'defect_end': {0: 'batch_size', 1: 'num_signals'},
        }
    )

    print(f"Model exported to {onnx_model_path}")

signal_length = 320
hidden_sizes = [128, 64, 32]
num_heads = 8
dropout = 0.2
num_transformer_layers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model = ImprovedMultiSignalClassifier(
# model = EnhancedPositionMultiSignalClassifier(
model = FixedEnhancedPositionMultiSignalClassifier(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers
    ).to(device)

modelname = "FixedEnhancedPositionMSC"
attempt = "000"
# model_path = f'models/improved_model_20250615_193609/best_model.pth'
# model_path = f'models/enhanced_position_model_20250711_1601/best_enhanced_position_model.pth'
model_path = f'models/fixed_enhanced_position_model_20250716_2032/best_fixed_enhanced_position_model.pth'
export_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length, hidden_sizes)
