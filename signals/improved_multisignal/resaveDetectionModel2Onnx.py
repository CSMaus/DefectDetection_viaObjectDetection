from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel

import torch
import torch.nn as nn

def create_onnx_compatible_complex_model(signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
    """Create ONNX-compatible version of ComplexDetectionModel by replacing adaptive pooling"""
    
    class ONNXComplexDetectionModel(nn.Module):
        def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
            super(ONNXComplexDetectionModel, self).__init__()
            
            # COMPLEX PREPROCESSING - Multi-scale 1D convolutions (same as original)
            self.conv_layers = nn.Sequential(
                # Local patterns (3-point)
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                
                # Medium patterns (7-point)
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                # Long patterns (15-point)
                nn.Conv1d(64, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # REPLACE ADAPTIVE POOLING WITH REGULAR POOLING
            # Calculate kernel size to get approximately 128 output
            kernel_size = signal_length // 128
            if kernel_size < 1:
                kernel_size = 1
            self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
            
            # Feature projection
            self.feature_projection = nn.Sequential(
                nn.Linear(128, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # STANDARD TRANSFORMER (same as original)
            self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # DETECTION HEAD
            self.detection_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        
        def forward(self, x):
            batch_size, num_signals, signal_length = x.size()
            
            # Reshape for 1D convolutions: (batch * num_signals, 1, signal_length)
            x = x.view(batch_size * num_signals, 1, signal_length)
            
            # Multi-scale feature extraction
            x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
            
            # Regular pooling instead of adaptive
            x = self.avg_pool(x)  # (batch * num_signals, 64, ~128)
            
            # Ensure we have exactly 128 features
            if x.size(2) != 128:
                # If not exactly 128, use interpolation to get 128
                x = torch.nn.functional.interpolate(x, size=128, mode='linear', align_corners=False)
            
            # Global average pooling across channels
            x = x.mean(dim=1)  # (batch * num_signals, 128)
            
            # Feature projection
            x = self.feature_projection(x)  # (batch * num_signals, d_model)
            
            # Reshape back: (batch, num_signals, d_model)
            x = x.view(batch_size, num_signals, -1)
            
            # Add positional encoding
            seq_len = x.size(1)
            pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
            x = x + pos_enc
            
            # Transformer processing
            x = self.transformer(x)
            
            # Detection prediction
            detection_logits = self.detection_head(x).squeeze(-1)
            detection_prob = torch.sigmoid(detection_logits)
            
            return detection_prob
    
    return ONNXComplexDetectionModel(signal_length, d_model, num_heads, num_layers, dropout)


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
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
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
    # Use ONNX-compatible version
    model = create_onnx_compatible_complex_model(signal_length=320).to(device)
    modelname = "ComplexDetectionModel"

attempt = "001"

model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'

export_detection_model_to_onnx(model, device, model_path,
                     f'models/{attempt}-{modelname}.onnx', signal_length)
