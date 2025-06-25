import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, num_signals, hidden_dim = x.shape
        position_encoding = self.encoding[:num_signals, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + position_encoding


class LocalAttention(nn.Module):
    """
    Local attention using convolutional layers to focus on neighboring signals.
    Increased kernel size for wider context window.
    """
    def __init__(self, d_model, kernel_size=9):  # Increased from 5 to 9 for wider context
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
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Local attention with residual connection and normalization
        local_attn_out = self.local_attn(x)
        x = self.norm2(x + self.dropout(local_attn_out))
        
        # Feed-forward network with residual connection and normalization
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class ImprovedMultiSignalClassifier(nn.Module):
    def __init__(self, signal_length, hidden_sizes, num_heads=8, dropout=0.1, num_transformer_layers=4):
        super(ImprovedMultiSignalClassifier, self).__init__()

        # 1D Convolutional feature extraction
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Background trend extraction with larger kernel for better context
        self.background_extractor = nn.Conv1d(
            in_channels=32, 
            out_channels=32, 
            kernel_size=15,  # Increased from 11 to 15
            padding=7, 
            stride=1,
            groups=32
        )

        # Feature extraction layers
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # Positional encoding and transformer layers
        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])  # Changed back to 300 as in original
        
        # Stack multiple transformer encoder layers (increased from original)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)  # Increased number of transformer layers
        ])
        
        # Output heads
        self.classifier = nn.Linear(hidden_sizes[1], 3)  # [defect_prob, start, end]

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Apply convolutional feature extraction
        x = self.conv1d(x)
        
        # Extract and remove background trend
        bg_trend = self.background_extractor(x)
        x = x - bg_trend  # Normalize signal based on extracted background
        
        # Global average pooling along the channel dimension
        x = x.mean(dim=1)  # shape becomes (batch_size*num_signals, signal_length)
        
        # Apply shared feature extraction
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)
        
        # Add positional encoding
        shared_out = self.position_encoding(shared_out)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            shared_out = transformer(shared_out)
        
        # Generate outputs
        outputs = self.classifier(shared_out)
        
        # Split and apply appropriate activations
        defect_prob = torch.sigmoid(outputs[:, :, 0])  # Keep sigmoid for binary classification
        
        # Direct regression for position prediction (no sigmoid)
        defect_start = outputs[:, :, 1]
        defect_end = outputs[:, :, 2]
        
        # Ensure positions are in [0,1] range using clamping instead of sigmoid
        defect_start = torch.clamp(defect_start, 0.0, 1.0)
        defect_end = torch.clamp(defect_end, 0.0, 1.0)
        
        return defect_prob, defect_start, defect_end

    def predict(self, x, threshold=0.5):
        """
        Make predictions with confidence threshold
        
        Args:
            x: Input tensor of shape [batch_size, num_signals, signal_length]
            threshold: Confidence threshold for defect detection
            
        Returns:
            List of dictionaries containing predictions for each batch
        """
        batch_size, num_signals, _ = x.shape
        defect_prob, defect_start, defect_end = self(x)
        
        results = []
        
        for b in range(batch_size):
            batch_results = []
            
            for i in range(num_signals):
                prob = defect_prob[b, i].item()
                
                if prob >= threshold:
                    batch_results.append({
                        'position': i,
                        'defect_prob': prob,
                        'defect_position': [
                            defect_start[b, i].item(),
                            defect_end[b, i].item()
                        ]
                    })
            
            results.append(batch_results)
        
        return results
