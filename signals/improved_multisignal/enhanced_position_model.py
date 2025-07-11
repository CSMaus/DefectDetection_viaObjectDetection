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
    """
    def __init__(self, d_model, kernel_size=9):
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


class EnhancedPositionMultiSignalClassifier(nn.Module):
    """
    Enhanced model with dedicated position regression head for much better localization accuracy
    """
    def __init__(self, signal_length, hidden_sizes, num_heads=8, dropout=0.1, num_transformer_layers=4):
        super(EnhancedPositionMultiSignalClassifier, self).__init__()

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

        # Enhanced background trend extraction
        self.background_extractor = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, padding=7, groups=32),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=31, padding=15, groups=32),  # Larger kernel for better trend extraction
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
        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        
        # Stack multiple transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # *** KEY CHANGE: Separate specialized heads ***
        
        # Detection head (binary classification)
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_sizes[1] // 2, 1)
        )
        
        # Position regression head (much more sophisticated)
        self.position_head = nn.Sequential(
            # First layer: extract position-relevant features
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Second layer: refine position features
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Third layer: position-specific processing
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.LayerNorm(hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            
            # Final layer: output positions
            nn.Linear(hidden_sizes[1] // 2, 2)  # [start, end]
        )
        
        # Multi-scale position prediction (additional head for robustness)
        self.position_head_coarse = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 4),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 4, 2)  # Coarse position prediction
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Apply convolutional feature extraction
        x = self.conv1d(x)
        
        # Extract and remove background trend (enhanced)
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
        
        # *** KEY CHANGE: Use specialized heads ***
        
        # Detection prediction
        detection_logits = self.detection_head(shared_out).squeeze(-1)
        defect_prob = torch.sigmoid(detection_logits)
        
        # Position prediction (fine-grained)
        position_fine = self.position_head(shared_out)
        
        # Position prediction (coarse-grained for robustness)
        position_coarse = self.position_head_coarse(shared_out)
        
        # Combine fine and coarse predictions (weighted average)
        position_combined = 0.7 * position_fine + 0.3 * position_coarse
        
        # Extract start and end positions
        defect_start = position_combined[:, :, 0]
        defect_end = position_combined[:, :, 1]
        
        # Enhanced position constraints
        # Ensure start < end and both in [0,1]
        defect_start = torch.sigmoid(defect_start)  # Use sigmoid for better gradient flow
        defect_end = torch.sigmoid(defect_end)
        
        # Ensure start <= end by swapping if necessary
        min_pos = torch.minimum(defect_start, defect_end)
        max_pos = torch.maximum(defect_start, defect_end)
        defect_start = min_pos
        defect_end = max_pos
        
        # Add small gap to prevent start == end
        gap = 0.01
        defect_end = torch.clamp(defect_end, min=defect_start + gap, max=1.0)
        
        return defect_prob, defect_start, defect_end

    def predict(self, x, threshold=0.5):
        """
        Make predictions with confidence threshold
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
    
    def freeze_detection_head(self):
        """Freeze detection head for position-focused training"""
        for param in self.detection_head.parameters():
            param.requires_grad = False
    
    def unfreeze_detection_head(self):
        """Unfreeze detection head"""
        for param in self.detection_head.parameters():
            param.requires_grad = True
    
    def freeze_position_head(self):
        """Freeze position head for detection-focused training"""
        for param in self.position_head.parameters():
            param.requires_grad = False
        for param in self.position_head_coarse.parameters():
            param.requires_grad = False
    
    def unfreeze_position_head(self):
        """Unfreeze position head"""
        for param in self.position_head.parameters():
            param.requires_grad = True
        for param in self.position_head_coarse.parameters():
            param.requires_grad = True
