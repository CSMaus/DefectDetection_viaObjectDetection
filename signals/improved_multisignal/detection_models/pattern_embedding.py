import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternEmbeddingModel(nn.Module):
    """
    Pattern embedding model for defect detection
    Maps signals with different fluctuation levels to similar background pattern embeddings
    Detects defects as deviations from expected pattern embedding space
    """
    def __init__(self, signal_length=320, d_model=64, num_heads=8, num_layers=4, dropout=0.1):
        super(PatternEmbeddingModel, self).__init__()
        
        # FEATURE EXTRACTION - convert signals to feature representations
        self.conv_layers = nn.Sequential(
            # Small kernels to preserve pattern details
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Capture local pattern structure
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Medium scale pattern features
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
        
        # PATTERN EMBEDDING PROJECTION
        # Maps raw features to pattern embedding space
        self.pattern_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # PATTERN EMBEDDING TRANSFORMER
        # Learns to create consistent embeddings for same background pattern
        # regardless of fluctuation level
        self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.pattern_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # BACKGROUND PATTERN REFERENCE
        # Learnable reference embedding for normal background pattern
        self.background_reference = nn.Parameter(torch.randn(1, d_model))
        
        # PATTERN SIMILARITY HEAD
        # Measures how similar each signal embedding is to background pattern
        self.similarity_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concat signal embedding + background reference
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        # Extract features from signals
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)
        
        # Fixed pooling
        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)
        
        # Ensure exactly 128 features
        current_size = x.size(2)
        if current_size != 128:
            x = F.interpolate(x, size=128, mode='linear', align_corners=False)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch * num_signals, 128)
        
        # Project to pattern embedding space
        x = self.pattern_projection(x)  # (batch * num_signals, d_model)
        
        # Reshape back for transformer
        x = x.view(batch_size, num_signals, -1)  # (batch, num_signals, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Pattern embedding transformer
        # Creates consistent embeddings for same background pattern
        pattern_embeddings = self.pattern_transformer(x)  # (batch, num_signals, d_model)
        
        # Expand background reference for all signals
        background_ref = self.background_reference.expand(batch_size, num_signals, -1)
        
        # Concatenate pattern embeddings with background reference
        combined = torch.cat([pattern_embeddings, background_ref], dim=-1)  # (batch, num_signals, d_model*2)
        
        # Calculate similarity to background pattern
        # High similarity = normal background (even with fluctuations)
        # Low similarity = defect (pattern break)
        similarity_logits = self.similarity_head(combined).squeeze(-1)  # (batch, num_signals)
        detection_prob = torch.sigmoid(similarity_logits)
        
        return detection_prob
