import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model)*0.2)

    def forward(self, x):
        batch_size, num_signals, hidden_dim = x.shape
        position_encoding = self.encoding[:num_signals, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + position_encoding


class LocalAttention(nn.Module):
    """
    Local attention using convolutional layers to focus on neighboring signals.
    Increased kernel size for wider context window.
    """
    def __init__(self, d_model, kernel_size=5):
        # Increased from 5 to kernel 9 for wider context
        super().__init__()
        self.local_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=d_model)

    def forward(self, x):
        # (batch, num_signals, d_model) -> (batch, d_model, num_signals)
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)  # Apply depth-wise convolution
        x = x.permute(0, 2, 1)  # Back to (batch, num_signals, d_model)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.local_attn = LocalAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and normalization
        # attn_out, _ = self.self_attn(x, x, x)
        # x = self.norm1(x + self.dropout(attn_out))
        xa = self.norm1(x)
        attn_out, _ = self.self_attn(xa, xa, xa)
        x = x + attn_out
        
        # Local attention with residual connection and normalization
        local_attn_out = self.local_attn(x)
        x = self.norm2(x + self.dropout(local_attn_out))
        
        # Feed-forward network with residual connection and normalization
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class HybridBinaryModel(nn.Module):
    """
    Hybrid model: improved_model transformer + direct_defect feature extraction
    Binary classification only (defect/no-defect)
    """
    def __init__(self, signal_length=320, hidden_sizes=[128, 64, 32], num_heads=8, dropout=0.1, num_transformer_layers=4):
        super(HybridBinaryModel, self).__init__()

        # FEATURE EXTRACTION from direct_defect.py
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

        # Feature extraction layers (similar to improved_model shared_layer)
        self.shared_layer = nn.Sequential(
            nn.Linear(128, hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # EXACT TRANSFORMER from improved_model.py
        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        
        # Stack multiple transformer encoder layers (from improved_model)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # BINARY CLASSIFICATION HEAD (only defect probability)
        self.classifier = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # FEATURE EXTRACTION from direct_defect.py
        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
        
        # Fixed pooling
        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)
        
        # Ensure exactly 128 features
        current_size = x.size(2)
        # if current_size != 128:
        x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch * num_signals, 128)
        
        # Apply shared feature extraction (from improved_model)
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)
        
        # Add positional encoding (from improved_model)
        shared_out = self.position_encoding(shared_out)
        
        # Apply transformer layers (EXACT from improved_model)
        for transformer in self.transformer_layers:
            shared_out = transformer(shared_out)
        
        # BINARY CLASSIFICATION (only defect probability)
        defect_logits = self.classifier(shared_out).squeeze(-1)  # (batch, num_signals)
        defect_prob = torch.sigmoid(defect_logits)
        
        return defect_prob
