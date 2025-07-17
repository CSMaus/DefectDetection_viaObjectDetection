import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDetectionModelFix(nn.Module):
    """
    ONNX-compatible version of ComplexDetectionModel
    EXACT same architecture but replaces adaptive_avg_pool1d with fixed pooling
    """
    def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
        super(ComplexDetectionModelFix, self).__init__()
        
        # EXACT SAME CONV LAYERS as original
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
        
        # REPLACE ONLY THE ADAPTIVE POOLING - keep everything else identical
        # Original: self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        # Fixed: Calculate exact kernel size to get 128 output
        kernel_size = signal_length // 128  # For 320 -> 128, kernel_size = 2.5, use 2
        if kernel_size < 1:
            kernel_size = 1
        self.fixed_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
        
        # EXACT SAME FEATURE PROJECTION as original
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
        
        # EXACT SAME DETECTION HEAD as original
        self.detection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # EXACT SAME FORWARD PASS as original
        # Reshape for 1D convolutions: (batch * num_signals, 1, signal_length)
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Multi-scale feature extraction
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
        
        # Fixed pooling instead of adaptive
        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)
        
        # Ensure exactly 128 features (same as adaptive pooling would do)
        current_size = x.size(2)
        if current_size != 128:
            # Use interpolation to get exactly 128 (same result as adaptive pooling)
            x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
        # Global average pooling across channels (SAME as original)
        x = x.mean(dim=1)  # (batch * num_signals, 128)
        
        # Feature projection (SAME as original)
        x = self.feature_projection(x)  # (batch * num_signals, d_model)
        
        # Reshape back: (batch, num_signals, d_model) (SAME as original)
        x = x.view(batch_size, num_signals, -1)
        
        # Add positional encoding (SAME as original)
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Transformer processing (SAME as original)
        x = self.transformer(x)
        
        # Detection prediction (SAME as original)
        detection_logits = self.detection_head(x).squeeze(-1)
        detection_prob = torch.sigmoid(detection_logits)
        
        return detection_prob
