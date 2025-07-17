import torch
import torch.nn as nn
import torch.nn.functional as F


class FluctuationRobustModel(nn.Module):
    """
    Model designed to handle background pattern fluctuations
    Distinguishes between background variations and true defects
    """
    def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
        super(FluctuationRobustModel, self).__init__()
        
        # SMALL KERNEL FEATURE EXTRACTION - avoid amplifying fluctuations
        self.conv_layers = nn.Sequential(
            # Start small to preserve defect details without amplifying fluctuations
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Keep small kernels - better for fluctuation handling
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Slightly larger only after initial processing
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ONNX-COMPATIBLE POOLING
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
        
        # ENHANCED TRANSFORMER - better pattern understanding
        self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=12,  # More attention heads for better pattern discrimination
            dim_feedforward=d_model * 3,  # Larger feedforward for complex pattern understanding
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)  # More layers for sequence understanding
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Small kernel feature extraction - preserve defect details
        x = self.conv_layers(x)
        
        # Fixed pooling
        x = self.fixed_pool(x)
        
        # Ensure exactly 128 features
        current_size = x.size(2)
        if current_size != 128:
            x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Feature projection
        x = self.feature_projection(x)
        
        # Reshape back for transformer
        x = x.view(batch_size, num_signals, -1)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Enhanced transformer - learns to distinguish fluctuations from defects
        x = self.transformer(x)
        
        # Detection prediction
        detection_logits = self.detection_head(x).squeeze(-1)
        detection_prob = torch.sigmoid(detection_logits)
        
        return detection_prob
