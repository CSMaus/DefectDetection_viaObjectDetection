import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexDetectionModel(nn.Module):
    """
    Detection model with complex preprocessing - multi-scale 1D conv feature extraction
    Standard transformer: current setup from your working model
    """
    def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
        super(ComplexDetectionModel, self).__init__()
        
        # COMPLEX PREPROCESSING - Multi-scale 1D convolutions
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
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # STANDARD TRANSFORMER (like your working model)
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
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch * num_signals, 64, 128)
        
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
