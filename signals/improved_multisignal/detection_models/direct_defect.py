import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectDefectModel(nn.Module):
    """
    Direct defect detection using sequence context
    Transformer processes all signals together and directly classifies each as defect/no-defect
    Uses attention across sequence for context, no comparisons
    """
    def __init__(self, signal_length=320, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
        super(DirectDefectModel, self).__init__()
        
        # FEATURE EXTRACTION from each signal
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
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
        
        # PROJECT TO TRANSFORMER DIMENSION
        self.feature_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # TRANSFORMER - processes all signals together with attention
        self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # DIRECT DEFECT CLASSIFICATION HEAD
        self.defect_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Extract features from each signal
        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
        
        # Fixed pooling
        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)
        
        # Ensure exactly 128 features
        current_size = x.size(2)
        if current_size != 128:
            x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch * num_signals, 128)
        
        # Project to transformer dimension
        x = self.feature_projection(x)  # (batch * num_signals, d_model)
        
        # Reshape for transformer: (batch, num_signals, d_model)
        x = x.view(batch_size, num_signals, -1)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # TRANSFORMER PROCESSES ALL SIGNALS TOGETHER
        # Attention provides sequence context automatically
        x = self.transformer(x)  # (batch, num_signals, d_model)
        
        # DIRECT DEFECT CLASSIFICATION
        # For each signal: defect or no-defect
        detection_logits = self.defect_classifier(x).squeeze(-1)  # (batch, num_signals)
        detection_prob = torch.sigmoid(detection_logits)
        
        return detection_prob
