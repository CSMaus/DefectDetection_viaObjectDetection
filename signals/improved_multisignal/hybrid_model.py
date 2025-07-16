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
    def __init__(self, d_model, kernel_size=9):
        super().__init__()
        self.local_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)
        x = x.permute(0, 2, 1)
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
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        local_attn_out = self.local_attn(x)
        x = self.norm2(x + self.dropout(local_attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class HybridModel(nn.Module):
    """
    Keeps your proven detection architecture (97% accuracy) and adds separate position module
    """
    def __init__(self, signal_length, hidden_sizes, num_heads=8, dropout=0.1, num_transformer_layers=4):
        super(HybridModel, self).__init__()

        # PROVEN DETECTION PATH (from your improved_model.py)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.background_extractor = nn.Conv1d(
            in_channels=32, 
            out_channels=32, 
            kernel_size=15,
            padding=7, 
            stride=1,
            groups=32
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # DETECTION HEAD (same as your working model)
        self.detection_classifier = nn.Linear(hidden_sizes[1], 1)
        
        # SEPARATE POSITION MODULE (doesn't interfere with detection)
        self.position_module = nn.Sequential(
            nn.Linear(hidden_sizes[1] + 1, hidden_sizes[1]),  # +1 for detection confidence
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
            nn.LayerNorm(hidden_sizes[1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(hidden_sizes[1] // 2, 2)  # [start, end]
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        
        # EXACT SAME PROCESSING AS YOUR WORKING MODEL
        x = x.view(batch_size * num_signals, 1, signal_length)
        
        x = self.conv1d(x)
        
        bg_trend = self.background_extractor(x)
        x = x - bg_trend
        
        x = x.mean(dim=1)
        
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)
        
        shared_out = self.position_encoding(shared_out)
        
        transformer_features = shared_out
        for transformer in self.transformer_layers:
            transformer_features = transformer(transformer_features)
        
        # DETECTION (same as your working model)
        detection_logits = self.detection_classifier(transformer_features).squeeze(-1)
        defect_prob = torch.sigmoid(detection_logits)
        
        # POSITION PREDICTION (uses detection confidence)
        position_input = torch.cat([
            transformer_features,
            defect_prob.unsqueeze(-1)
        ], dim=-1)
        
        position_outputs = self.position_module(position_input)
        defect_start = torch.sigmoid(position_outputs[:, :, 0])
        defect_end = torch.sigmoid(position_outputs[:, :, 1])
        
        # Ensure start <= end
        min_pos = torch.minimum(defect_start, defect_end)
        max_pos = torch.maximum(defect_start, defect_end)
        defect_start = min_pos
        defect_end = max_pos
        
        gap = 0.01
        defect_end = torch.maximum(defect_end, defect_start + gap)
        defect_end = torch.clamp(defect_end, max=1.0)
        
        return defect_prob, defect_start, defect_end

    def freeze_detection_path(self):
        """Freeze the proven detection path"""
        for param in self.conv1d.parameters():
            param.requires_grad = False
        for param in self.background_extractor.parameters():
            param.requires_grad = False
        for param in self.shared_layer.parameters():
            param.requires_grad = False
        for param in self.position_encoding.parameters():
            param.requires_grad = False
        for param in self.transformer_layers.parameters():
            param.requires_grad = False
        for param in self.detection_classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_detection_path(self):
        """Unfreeze the detection path"""
        for param in self.conv1d.parameters():
            param.requires_grad = True
        for param in self.background_extractor.parameters():
            param.requires_grad = True
        for param in self.shared_layer.parameters():
            param.requires_grad = True
        for param in self.position_encoding.parameters():
            param.requires_grad = True
        for param in self.transformer_layers.parameters():
            param.requires_grad = True
        for param in self.detection_classifier.parameters():
            param.requires_grad = True
    
    def freeze_position_module(self):
        """Freeze only the position module"""
        for param in self.position_module.parameters():
            param.requires_grad = False
    
    def unfreeze_position_module(self):
        """Unfreeze only the position module"""
        for param in self.position_module.parameters():
            param.requires_grad = True
