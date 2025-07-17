from detection_models.complex_detection_model import ComplexDetectionModel
import torch
import torch.nn as nn

def make_onnx_compatible_model(original_model, device):
    """Replace adaptive pooling with fixed pooling to make ONNX compatible"""
    
    # Create new model with same architecture but ONNX-compatible pooling
    class ONNXComplexDetectionModel(nn.Module):
        def __init__(self, original_model, device):
            super().__init__()
            
            # Copy all layers from original model
            self.conv_layers = original_model.conv_layers
            
            # Replace adaptive pooling with fixed operations
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # This should work in ONNX
            self.feature_expand = nn.Linear(64, 128).to(device)  # Ensure on correct device
            
            self.feature_projection = original_model.feature_projection
            self.positional_encoding = original_model.positional_encoding
            self.transformer = original_model.transformer
            self.detection_head = original_model.detection_head
        
        def forward(self, x):
            batch_size, num_signals, signal_length = x.size()
            
            # Reshape for 1D convolutions
            x = x.view(batch_size * num_signals, 1, signal_length)
            
            # Multi-scale feature extraction
            x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
            
            # Global average pooling to single value per channel
            x = self.global_avg_pool(x)  # (batch * num_signals, 64, 1)
            x = x.squeeze(-1)  # (batch * num_signals, 64)
            
            # Expand to 128 features using linear layer
            x = self.feature_expand(x)  # (batch * num_signals, 128)
            
            # Feature projection
            x = self.feature_projection(x)  # (batch * num_signals, d_model)
            
            # Reshape back
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
    
    return ONNXComplexDetectionModel(original_model, device)

def export_model_to_onnx(model_path, onnx_model_path, signal_length=320):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load original model
    original_model = ComplexDetectionModel(signal_length=signal_length).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        original_model.load_state_dict(checkpoint)
        print("Loaded checkpoint directly")
    
    # Create ONNX-compatible version
    onnx_model = make_onnx_compatible_model(original_model, device)
    onnx_model.to(device)  # Ensure entire model is on device
    onnx_model.eval()
    
    # Initialize the new linear layer with reasonable weights
    with torch.no_grad():
        # Initialize feature_expand layer
        nn.init.xavier_uniform_(onnx_model.feature_expand.weight)
        nn.init.zeros_(onnx_model.feature_expand.bias)
    
    # Create dummy input
    dummy_input = torch.randn(1, 50, signal_length).to(device)
    
    # Test forward pass
    with torch.no_grad():
        output = onnx_model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=15,  # Use your opset 15
        input_names=['input'],
        output_names=['detection_prob']
    )
    
    print(f"Model exported to {onnx_model_path}")

# Main execution
modelname = "ComplexDetectionModel"
attempt = "000"
model_path = f'models/Complex_20250717_0800/best_complex_detection.pth'

export_model_to_onnx(model_path, f'models/{attempt}-{modelname}.onnx')
