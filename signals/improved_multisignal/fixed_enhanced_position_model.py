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


class FixedEnhancedPositionMultiSignalClassifier(nn.Module):
    """
    FIXED: Truly separate detection and position heads with independent processing paths
    """
    def __init__(self, signal_length, hidden_sizes, num_heads=8, dropout=0.1, num_transformer_layers=4):
        super(FixedEnhancedPositionMultiSignalClassifier, self).__init__()

        # SHARED: 1D Convolutional feature extraction (common for both tasks)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # SHARED: Enhanced background trend extraction
        self.background_extractor = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, padding=7, groups=32),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=31, padding=15, groups=32),
        )

        # SHARED: Initial feature extraction (common base)
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # SHARED: Positional encoding
        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        
        # *** SEPARATE TRANSFORMER STACKS FOR EACH TASK ***
        
        # Detection-specific transformer layers
        self.detection_transformers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Position-specific transformer layers (completely separate)
        self.position_transformers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # *** DETECTION HEAD: Uses only detection transformer features ***
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_sizes[1] // 2, 1)
        )
        
        # *** POSITION HEAD: Uses only position transformer features + original features ***
        # Input: [original_features + position_transformer_features]
        position_input_dim = hidden_sizes[1] + hidden_sizes[1]  # shared + position_transformer
        
        self.position_head = nn.Sequential(
            # First layer: process combined features
            nn.Linear(position_input_dim, hidden_sizes[1]),
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
        
        # Multi-scale position prediction (coarse head)
        self.position_head_coarse = nn.Sequential(
            nn.Linear(position_input_dim, hidden_sizes[1] // 4),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1] // 4, 2)  # Coarse position prediction
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Apply convolutional feature extraction (SHARED)
        x = self.conv1d(x)
        
        # Extract and remove background trend (SHARED)
        bg_trend = self.background_extractor(x)
        x = x - bg_trend
        
        # Global average pooling along the channel dimension
        x = x.mean(dim=1)  # shape becomes (batch_size*num_signals, signal_length)
        
        # Apply shared feature extraction
        shared_features = self.shared_layer(x)
        shared_features = shared_features.view(batch_size, num_signals, -1)
        
        # Add positional encoding (SHARED)
        shared_features_with_pos = self.position_encoding(shared_features)
        
        # *** SEPARATE PROCESSING PATHS ***
        
        # Detection path: Use detection-specific transformers
        detection_features = shared_features_with_pos
        for transformer in self.detection_transformers:
            detection_features = transformer(detection_features)
        
        # Position path: Use position-specific transformers
        position_features = shared_features_with_pos
        for transformer in self.position_transformers:
            position_features = transformer(position_features)
        
        # *** DETECTION HEAD: Uses ONLY detection transformer features ***
        detection_logits = self.detection_head(detection_features).squeeze(-1)
        defect_prob = torch.sigmoid(detection_logits)
        
        # *** POSITION HEAD: Uses ONLY position transformer features + original shared features ***
        # NO dependency on detection features!
        position_input = torch.cat([
            shared_features,  # Original features (before transformers)
            position_features,  # Position-optimized features
        ], dim=-1)
        
        # Position prediction (fine-grained)
        position_fine = self.position_head(position_input)
        
        # Position prediction (coarse-grained for robustness)  
        position_coarse = self.position_head_coarse(position_input)
        
        # Combine fine and coarse predictions (weighted average)
        position_combined = 0.7 * position_fine + 0.3 * position_coarse
        
        # Extract start and end positions
        defect_start = position_combined[:, :, 0]
        defect_end = position_combined[:, :, 1]
        
        # Enhanced position constraints
        # Ensure start < end and both in [0,1]
        defect_start = torch.sigmoid(defect_start)
        defect_end = torch.sigmoid(defect_end)
        
        # Ensure start <= end by swapping if necessary
        min_pos = torch.minimum(defect_start, defect_end)
        max_pos = torch.maximum(defect_start, defect_end)
        defect_start = min_pos
        defect_end = max_pos
        
        # Add small gap to prevent start == end
        gap = 0.01
        defect_end = torch.maximum(defect_end, defect_start + gap)
        defect_end = torch.clamp(defect_end, max=1.0)
        
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
    
    def freeze_detection_path(self):
        """Freeze entire detection path for position-focused training"""
        for param in self.detection_transformers.parameters():
            param.requires_grad = False
        for param in self.detection_head.parameters():
            param.requires_grad = False
    
    def unfreeze_detection_path(self):
        """Unfreeze entire detection path"""
        for param in self.detection_transformers.parameters():
            param.requires_grad = True
        for param in self.detection_head.parameters():
            param.requires_grad = True
    
    def freeze_position_path(self):
        """Freeze entire position path for detection-focused training"""
        for param in self.position_transformers.parameters():
            param.requires_grad = False
        for param in self.position_head.parameters():
            param.requires_grad = False
        for param in self.position_head_coarse.parameters():
            param.requires_grad = False
    
    def unfreeze_position_path(self):
        """Unfreeze entire position path"""
        for param in self.position_transformers.parameters():
            param.requires_grad = True
        for param in self.position_head.parameters():
            param.requires_grad = True
        for param in self.position_head_coarse.parameters():
            param.requires_grad = True
