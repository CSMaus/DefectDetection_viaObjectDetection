import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedPatternModel(nn.Module):
    """
    Enhanced pattern embedding model with contrastive learning approach
    Multiple background prototypes + explicit defect discrimination
    Fixes both false positives (background as defects) and false negatives (missed defects)
    """
    def __init__(self, signal_length=320, d_model=128, num_heads=16, num_layers=8, dropout=0.1, num_bg_prototypes=5):
        super(EnhancedPatternModel, self).__init__()
        
        self.d_model = d_model
        self.num_bg_prototypes = num_bg_prototypes
        
        # FEATURE EXTRACTION - same approach but feeding larger model
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
        
        # ENHANCED PATTERN EMBEDDING PROJECTION
        self.pattern_projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ENHANCED PATTERN EMBEDDING TRANSFORMER
        self.positional_encoding = nn.Parameter(torch.randn(300, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            batch_first=True
        )
        self.pattern_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MULTIPLE BACKGROUND PROTOTYPES - capture different background pattern variations
        self.background_prototypes = nn.Parameter(torch.randn(num_bg_prototypes, d_model))
        
        # BACKGROUND SIMILARITY HEAD - measures similarity to background prototypes
        self.background_similarity = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_bg_prototypes)  # Similarity to each prototype
        )
        
        # DEFECT DISCRIMINATOR HEAD - explicitly learns defect patterns
        self.defect_discriminator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # CONTRASTIVE FUSION HEAD - combines background similarity + defect discrimination
        self.fusion_head = nn.Sequential(
            nn.Linear(num_bg_prototypes + 1, d_model // 4),  # bg similarities + defect score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
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
        
        # Enhanced pattern embedding projection
        x = self.pattern_projection(x)  # (batch * num_signals, d_model)
        
        # Reshape back for transformer
        x = x.view(batch_size, num_signals, -1)  # (batch, num_signals, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Enhanced pattern embedding transformer
        pattern_embeddings = self.pattern_transformer(x)  # (batch, num_signals, d_model)
        
        # Flatten for processing
        flat_embeddings = pattern_embeddings.view(batch_size * num_signals, -1)  # (batch * num_signals, d_model)
        
        # BACKGROUND SIMILARITY COMPUTATION
        # Calculate similarity to each background prototype
        bg_similarities = self.background_similarity(flat_embeddings)  # (batch * num_signals, num_bg_prototypes)
        bg_similarities = torch.softmax(bg_similarities, dim=-1)  # Normalize similarities
        
        # DEFECT DISCRIMINATION
        # Explicit defect pattern recognition
        defect_scores = self.defect_discriminator(flat_embeddings)  # (batch * num_signals, 1)
        defect_scores = torch.sigmoid(defect_scores)
        
        # CONTRASTIVE FUSION
        # Combine background similarity + defect discrimination
        combined_features = torch.cat([bg_similarities, defect_scores], dim=-1)  # (batch * num_signals, num_bg_prototypes + 1)
        
        # Final detection decision
        detection_logits = self.fusion_head(combined_features).squeeze(-1)  # (batch * num_signals,)
        detection_prob = torch.sigmoid(detection_logits)
        
        # Reshape back to original format
        detection_prob = detection_prob.view(batch_size, num_signals)  # (batch, num_signals)
        
        return detection_prob
