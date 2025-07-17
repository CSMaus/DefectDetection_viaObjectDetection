import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseRobustDetectionModel(nn.Module):
    """
    Noise-robust ONNX-compatible detection model
    Enhanced feature extraction for highly noised data + exact same transformer
    """
    def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
        super(NoiseRobustDetectionModel, self).__init__()
        
        # NOISE-ROBUST FEATURE EXTRACTION - ONNX compatible
        self.conv_layers = nn.Sequential(
            # Pre-denoising layer with larger kernel for smoothing
            nn.Conv1d(1, 16, kernel_size=7, padding=3),  # Larger kernel for noise reduction
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Local patterns (5-point instead of 3 for better noise handling)
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Medium patterns (9-point instead of 7 for better noise handling)
            nn.Conv1d(32, 48, kernel_size=9, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Long patterns (17-point instead of 15 for better noise handling)
            nn.Conv1d(48, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Additional smoothing layer
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ONNX-COMPATIBLE POOLING - Fixed size pooling instead of adaptive
        kernel_size = signal_length // 128
        if kernel_size < 1:
            kernel_size = 1
        self.fixed_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # EXACT SAME TRANSFORMER as original
        self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Detection head
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
        
        # Noise-robust multi-scale feature extraction
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
        
        # Fixed pooling instead of adaptive
        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)
        
        # Ensure exactly 128 features
        current_size = x.size(2)
        if current_size != 128:
            x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
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
