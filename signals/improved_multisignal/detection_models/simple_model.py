import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDetectionModel(nn.Module):
    """
    Detection model with minimal preprocessing - let transformer do the work
    Enhanced transformer: more layers, more heads, longer sequences
    """
    def __init__(self, signal_length=320, d_model=128, num_heads=16, num_layers=8, dropout=0.1):
        super(SimpleDetectionModel, self).__init__()
        
        # MINIMAL PREPROCESSING
        self.input_projection = nn.Linear(signal_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # ENHANCED TRANSFORMER
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))  # Support longer sequences
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,  # More attention heads
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Deeper network
        
        # DETECTION HEAD
        self.detection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape: (batch * num_signals, signal_length)
        x = x.view(batch_size * num_signals, signal_length)
        
        # Simple projection to transformer dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Reshape back: (batch, num_signals, d_model)
        x = x.view(batch_size, num_signals, -1)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Enhanced transformer processing
        x = self.transformer(x)
        
        # Detection prediction
        detection_logits = self.detection_head(x).squeeze(-1)
        detection_prob = torch.sigmoid(detection_logits)
        
        return detection_prob
