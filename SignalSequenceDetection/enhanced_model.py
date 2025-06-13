import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D signals with dilated convolutions for multi-scale processing.
    """
    def __init__(self, channels, dilation=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return self.relu(out)


class MultiScaleModule(nn.Module):
    """
    Multi-scale processing module using dilated convolutions.
    This doesn't modify input data, but allows the model to analyze patterns
    at different scales simultaneously.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleModule, self).__init__()
        
        # Different dilation rates for multi-scale processing
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4)
        self.branch4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=8, dilation=8)
        
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return self.combine(outputs)


class EnhancedSignalEncoder(nn.Module):
    """
    Enhanced encoder for individual signals with residual connections and multi-scale processing.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=256):
        super(EnhancedSignalEncoder, self).__init__()
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-scale processing
        self.multi_scale = MultiScaleModule(hidden_dim, hidden_dim*2)
        
        # Residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim*2, dilation=1),
            ResidualBlock1D(hidden_dim*2, dilation=2),
            ResidualBlock1D(hidden_dim*2, dilation=4)
        ])
        
        # Feature pyramid for capturing features at different resolutions
        self.pyramid_1 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1)
        self.pyramid_bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.pyramid_2 = nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=3, stride=2, padding=1)
        self.pyramid_bn2 = nn.BatchNorm1d(hidden_dim*4)
        
        # Adaptive pooling to handle variable length inputs
        self.adaptive_pool_1 = nn.AdaptiveAvgPool1d(1)
        self.adaptive_pool_2 = nn.AdaptiveAvgPool1d(1)
        self.adaptive_pool_3 = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim*4 + hidden_dim*4, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
        Returns:
            Tensor, shape [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, signal_length = x.shape
        
        # Reshape for 1D convolution
        x = x.view(batch_size * seq_len, 1, signal_length)
        
        # Initial convolution
        x = self.conv_init(x)
        
        # Multi-scale processing
        x = self.multi_scale(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Extract features at different scales
        feat_orig = self.adaptive_pool_1(x).squeeze(-1)  # [batch*seq, hidden_dim*2]
        
        x1 = F.relu(self.pyramid_bn1(self.pyramid_1(x)))
        feat_mid = self.adaptive_pool_2(x1).squeeze(-1)  # [batch*seq, hidden_dim*4]
        
        x2 = F.relu(self.pyramid_bn2(self.pyramid_2(x1)))
        feat_small = self.adaptive_pool_3(x2).squeeze(-1)  # [batch*seq, hidden_dim*4]
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([feat_orig, feat_mid, feat_small], dim=1)
        
        # Final projection
        output = self.fc(multi_scale_features)
        
        # Reshape back to sequence
        output = output.view(batch_size, seq_len, -1)
        
        return output


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for sequence processing.
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu  # Using GELU activation (from transformers)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2, attention_weights = self.self_attn(src, src, src, 
                                                attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attention_weights


class EnhancedSequenceTransformer(nn.Module):
    """
    Enhanced transformer for processing sequences with multi-head attention and feed-forward networks.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(EnhancedSequenceTransformer, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer layers with attention weights output
        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output: Tensor, shape [batch_size, seq_len, d_model]
            attention_weights: List of attention weights from each layer
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, src_mask=mask)
            attention_weights.append(attn)
        
        # Apply final normalization
        output = self.norm(x)
        
        return output, attention_weights


class EnhancedContextAggregator(nn.Module):
    """
    Enhanced context aggregator with bidirectional LSTM and attention mechanism.
    """
    def __init__(self, d_model=256):
        super(EnhancedContextAggregator, self).__init__()
        
        # Bidirectional LSTM for context aggregation
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention_query = nn.Parameter(torch.randn(d_model))
        self.attention_keys = nn.Linear(d_model, d_model)
        self.attention_values = nn.Linear(d_model, d_model)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        # Compute attention scores
        keys = self.attention_keys(lstm_out)  # [batch_size, seq_len, d_model]
        query = self.attention_query.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)  # [batch_size, seq_len, d_model]
        
        # Compute attention scores and weights
        attention_scores = torch.sum(keys * query, dim=-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply attention to values
        values = self.attention_values(lstm_out)  # [batch_size, seq_len, d_model]
        weighted_values = values * attention_weights  # [batch_size, seq_len, d_model]
        
        # Concatenate original and weighted values
        context = torch.cat([lstm_out, weighted_values], dim=-1)  # [batch_size, seq_len, d_model*2]
        
        # Project back to original dimension
        output = self.projection(context)  # [batch_size, seq_len, d_model]
        
        return output, attention_weights.squeeze(-1)


class EnhancedAnomalyDetector(nn.Module):
    """
    Enhanced anomaly detector with contrastive learning approach.
    """
    def __init__(self, d_model=256, hidden_dim=128):
        super(EnhancedAnomalyDetector, self).__init__()
        
        # Health feature extractor
        self.health_extractor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, d_model)
        )
        
        # Anomaly detection network
        self.anomaly_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        
    def forward(self, signal_features, sequence_features):
        """
        Args:
            signal_features: Tensor, shape [batch_size, seq_len, d_model]
            sequence_features: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            anomaly_scores: Tensor, shape [batch_size, seq_len, 1]
            uncertainty: Tensor, shape [batch_size, seq_len, 1]
            health_features: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Extract health features from sequence context
        health_features = self.health_extractor(sequence_features)
        
        # Compute similarity between signal and health features
        combined = torch.cat([signal_features, health_features], dim=-1)
        
        # Compute anomaly scores
        anomaly_scores = self.anomaly_net(combined)
        
        # Compute uncertainty
        uncertainty = self.uncertainty_net(combined)
        
        return anomaly_scores, uncertainty, health_features


class EnhancedDefectDetectionHead(nn.Module):
    """
    Enhanced detection head for predicting defect class and position within 1D signals.
    """
    def __init__(self, d_model=256, num_classes=2):
        super(EnhancedDefectDetectionHead, self).__init__()
        
        # Class prediction with uncertainty
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Class uncertainty estimation
        self.class_uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_classes),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        
        # Defect position prediction (start, end) within the 1D signal
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Sigmoid()  # Normalize to [0, 1] range of the signal length
        )
        
        # Position uncertainty estimation
        self.position_uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            class_logits: Tensor, shape [batch_size, seq_len, num_classes]
            class_uncertainty: Tensor, shape [batch_size, seq_len, num_classes]
            position_pred: Tensor, shape [batch_size, seq_len, 2] (start, end positions in signal)
            position_uncertainty: Tensor, shape [batch_size, seq_len, 2]
        """
        class_logits = self.class_head(x)
        class_uncertainty = self.class_uncertainty(x)
        position_pred = self.position_head(x)
        position_uncertainty = self.position_uncertainty(x)
        
        return class_logits, class_uncertainty, position_pred, position_uncertainty
class EnhancedSignalSequenceDetector(nn.Module):
    """
    Enhanced model for detecting defects in sequences of signals.
    """
    def __init__(
        self,
        signal_length=100,
        d_model=256,
        num_classes=2,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super(EnhancedSignalSequenceDetector, self).__init__()
        
        # Signal encoder with multi-scale processing
        self.signal_encoder = EnhancedSignalEncoder(
            input_dim=1,
            hidden_dim=64,
            output_dim=d_model
        )
        
        # Sequence transformer with attention
        self.sequence_transformer = EnhancedSequenceTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Context aggregator
        self.context_aggregator = EnhancedContextAggregator(d_model=d_model)
        
        # Anomaly detector with uncertainty estimation
        self.anomaly_detector = EnhancedAnomalyDetector(d_model=d_model, hidden_dim=128)
        
        # Detection head with uncertainty estimation
        self.detection_head = EnhancedDefectDetectionHead(d_model=d_model, num_classes=num_classes)
        
        # Cross-attention for sequence-level context
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Sequence-level feature integration
        self.sequence_integration = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, targets=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
            targets: Dictionary or list with target information
        Returns:
            If targets is None:
                dict with predictions
            Else:
                loss, dict with loss components
        """
        batch_size, seq_len, signal_length = x.shape
        
        # 1. Encode individual signals with multi-scale processing
        signal_features = self.signal_encoder(x)
        
        # 2. Apply sequence transformer to capture temporal relationships
        sequence_features, attention_weights = self.sequence_transformer(signal_features)
        
        # 3. Aggregate context information with enhanced context aggregator
        context_features, context_attention = self.context_aggregator(sequence_features)
        
        # 4. Apply cross-attention between signal features and context
        cross_out, cross_attention = self.cross_attention(
            signal_features, context_features, context_features
        )
        cross_features = self.cross_norm(signal_features + cross_out)
        
        # 5. Integrate sequence-level features
        integrated_features = self.sequence_integration(
            torch.cat([cross_features, sequence_features], dim=-1)
        )
        
        # 6. Detect anomalies with uncertainty estimation
        anomaly_scores, anomaly_uncertainty, health_features = self.anomaly_detector(
            integrated_features, sequence_features
        )
        
        # 7. Predict class and defect positions with uncertainty
        class_logits, class_uncertainty, position_pred, position_uncertainty = self.detection_head(integrated_features)
        
        # 8. Enhance class predictions with anomaly scores
        # For non-health classes (index > 0), add anomaly score
        if class_logits.size(-1) > 1:  # More than just health/non-health
            # Clone to avoid in-place modification
            enhanced_logits = class_logits.clone()
            
            # Add anomaly scores to non-health classes
            enhanced_logits[:, :, 1:] = enhanced_logits[:, :, 1:] + anomaly_scores
            
            class_logits = enhanced_logits
        
        # 9. Return predictions or compute loss
        if targets is None:
            return {
                'class_preds': class_logits,
                'class_uncertainty': class_uncertainty,
                'position_preds': position_pred,
                'position_uncertainty': position_uncertainty,
                'anomaly_scores': anomaly_scores,
                'anomaly_uncertainty': anomaly_uncertainty,
                'attention_weights': attention_weights,
                'context_attention': context_attention,
                'cross_attention': cross_attention
            }
        
        # 10. Compute loss with uncertainty weighting
        loss_dict = self._compute_loss(
            class_logits, class_uncertainty,
            position_pred, position_uncertainty,
            anomaly_scores, anomaly_uncertainty,
            targets
        )
        
        total_loss = (
            loss_dict['cls_loss'] + 
            loss_dict['position_loss'] + 
            0.1 * loss_dict['anomaly_consistency_loss'] +
            0.05 * loss_dict['uncertainty_loss']
        )
        
        return total_loss, loss_dict
    
    def _compute_loss(self, class_logits, class_uncertainty, position_pred, position_uncertainty, 
                     anomaly_scores, anomaly_uncertainty, targets):
        """
        Compute loss for training with uncertainty weighting.
        
        Args:
            class_logits: Tensor, shape [batch_size, seq_len, num_classes]
            class_uncertainty: Tensor, shape [batch_size, seq_len, num_classes]
            position_pred: Tensor, shape [batch_size, seq_len, 2]
            position_uncertainty: Tensor, shape [batch_size, seq_len, 2]
            anomaly_scores: Tensor, shape [batch_size, seq_len, 1]
            anomaly_uncertainty: Tensor, shape [batch_size, seq_len, 1]
            targets: Dictionary with targets information
        
        Returns:
            dict with loss components
        """
        batch_size, seq_len, num_classes = class_logits.shape
        device = class_logits.device
        
        # Prepare target tensors
        target_classes = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        target_positions = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=device)
        
        # Check the structure of targets and adapt accordingly
        if isinstance(targets, list):
            # Handle list of dictionaries (one dict per batch item)
            for b in range(batch_size):
                if b < len(targets):
                    batch_targets = targets[b]
                    
                    # Handle different target structures
                    if isinstance(batch_targets, list):
                        # List of targets for this batch item
                        for i, target in enumerate(batch_targets):
                            if i < seq_len:
                                self._process_single_target(target, target_classes, target_positions, b, i)
                    elif isinstance(batch_targets, dict):
                        # Single target dictionary with multiple entries
                        if 'targets' in batch_targets and isinstance(batch_targets['targets'], list):
                            for i, target in enumerate(batch_targets['targets']):
                                if i < seq_len:
                                    self._process_single_target(target, target_classes, target_positions, b, i)
        elif isinstance(targets, dict):
            # Handle dictionary with 'targets' key
            if 'targets' in targets and isinstance(targets['targets'], list):
                for i, target in enumerate(targets['targets']):
                    if i < seq_len:
                        self._process_single_target(target, target_classes, target_positions, 0, i)
        
        # Classification loss - standard cross entropy
        flat_logits = class_logits.view(-1, num_classes)
        flat_targets = target_classes.view(-1)
        
        # Standard classification loss
        cls_loss = F.cross_entropy(flat_logits, flat_targets)
        
        # Add uncertainty regularization for classification
        # Use absolute value to ensure positive loss
        cls_uncertainty_reg = torch.mean(torch.exp(-class_uncertainty) + class_uncertainty)
        
        # Combined classification loss (always positive)
        weighted_cls_loss = cls_loss + 0.1 * cls_uncertainty_reg
        
        # Position prediction loss (only for defect signals)
        positive_mask = (target_classes > 0).unsqueeze(-1).expand_as(target_positions)
        
        if positive_mask.sum() > 0:
            # Standard L1 loss for defect positions
            pos_loss = F.l1_loss(
                position_pred[positive_mask],
                target_positions[positive_mask]
            )
            
            # Add uncertainty regularization for position
            pos_uncertainty_reg = torch.mean(torch.exp(-position_uncertainty[positive_mask]) + position_uncertainty[positive_mask])
            
            # Combined position loss (always positive)
            weighted_pos_loss = pos_loss + 0.1 * pos_uncertainty_reg
        else:
            weighted_pos_loss = torch.tensor(0.0, device=device)
        
        # Anomaly consistency loss
        anomaly_consistency_loss = 0.0
        
        if seq_len > 1:
            # Calculate temporal consistency loss for anomaly scores
            temporal_diffs = []
            for t in range(1, seq_len):
                diff = torch.abs(anomaly_scores[:, t] - anomaly_scores[:, t-1])
                temporal_diffs.append(diff)
            
            if temporal_diffs:
                # Use standard L1 loss for temporal consistency
                anomaly_consistency_loss = torch.cat(temporal_diffs).mean()
            
            # Add uncertainty regularization (always positive)
            uncertainty_reg = (
                torch.mean(torch.exp(-class_uncertainty) + class_uncertainty) + 
                torch.mean(torch.exp(-position_uncertainty) + position_uncertainty) + 
                torch.mean(torch.exp(-anomaly_uncertainty) + anomaly_uncertainty)
            ) * 0.01
        else:
            uncertainty_reg = torch.tensor(0.0, device=device)
        
        # Ensure all losses are positive
        total_loss = weighted_cls_loss + weighted_pos_loss + 0.1 * anomaly_consistency_loss + 0.05 * uncertainty_reg
        
        return {
            'cls_loss': weighted_cls_loss,
            'position_loss': weighted_pos_loss,
            'anomaly_consistency_loss': anomaly_consistency_loss,
            'uncertainty_loss': uncertainty_reg,
            'total_loss': total_loss
        }
    
    def _process_single_target(self, target, target_classes, target_positions, batch_idx, seq_idx):
        """
        Process a single target and update the target tensors.
        
        Args:
            target: Dictionary with target information
            target_classes: Tensor to update with class information
            target_positions: Tensor to update with position information
            batch_idx: Batch index
            seq_idx: Sequence index
        """
        # Process class label
        if 'label' in target:
            if isinstance(target['label'], str):
                # Default to 0 (assuming 0 is the first class)
                target_classes[batch_idx, seq_idx] = 0
            else:
                target_classes[batch_idx, seq_idx] = target['label']
        
        # Process defect position
        if target_classes[batch_idx, seq_idx] > 0:  # If it's a defect (not "Health")
            if 'defect_position' in target:
                if torch.is_tensor(target['defect_position']):
                    if target['defect_position'].numel() >= 2:
                        target_positions[batch_idx, seq_idx, 0] = target['defect_position'][0]
                        target_positions[batch_idx, seq_idx, 1] = target['defect_position'][1]
                else:
                    # Handle case where defect_position might be a list or numpy array
                    target_positions[batch_idx, seq_idx, 0] = target['defect_position'][0]
                    target_positions[batch_idx, seq_idx, 1] = target['defect_position'][1]
            elif 'bbox' in target:
                # Backward compatibility with old format
                if torch.is_tensor(target['bbox']):
                    if target['bbox'].numel() >= 4:
                        target_positions[batch_idx, seq_idx, 0] = target['bbox'][2]
                        target_positions[batch_idx, seq_idx, 1] = target['bbox'][3]
                else:
                    target_positions[batch_idx, seq_idx, 0] = target['bbox'][2]
                    target_positions[batch_idx, seq_idx, 1] = target['bbox'][3]
    
    def predict(self, x, threshold=0.5):
        """
        Make predictions with the model.
        
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
            threshold: Float, threshold for class prediction
            
        Returns:
            List of dicts with predictions for each sequence
        """
        # Get raw predictions
        with torch.no_grad():
            preds = self(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        class_preds = preds['class_preds']
        class_uncertainty = preds['class_uncertainty']
        position_preds = preds['position_preds']
        position_uncertainty = preds['position_uncertainty']
        anomaly_scores = preds['anomaly_scores']
        anomaly_uncertainty = preds['anomaly_uncertainty']
        
        results = []
        
        for b in range(batch_size):
            sequence_results = []
            
            for i in range(seq_len):
                # Get class probabilities
                class_probs = F.softmax(class_preds[b, i], dim=0)
                
                # Get predicted class (argmax)
                pred_class = torch.argmax(class_probs).item()
                class_score = class_probs[pred_class].item()
                
                # Get class uncertainty
                class_uncert = class_uncertainty[b, i, pred_class].item()
                
                # Get defect position within the signal
                pred_position = position_preds[b, i].cpu().numpy()
                pos_uncert = position_uncertainty[b, i].cpu().numpy()
                
                # Get anomaly score and uncertainty
                anomaly_score = anomaly_scores[b, i, 0].item()
                anom_uncert = anomaly_uncertainty[b, i, 0].item()
                
                # Adjust confidence based on uncertainty
                adjusted_confidence = class_score / (1.0 + class_uncert)
                
                # Add to results if confidence is above threshold
                if adjusted_confidence > threshold or (pred_class > 0 and anomaly_score > threshold):
                    sequence_results.append({
                        'position': i,  # Position in sequence
                        'class': pred_class,
                        'class_score': class_score,
                        'class_uncertainty': class_uncert,
                        'defect_position': pred_position,  # Start and end positions within the signal
                        'position_uncertainty': pos_uncert,
                        'anomaly_score': anomaly_score,
                        'anomaly_uncertainty': anom_uncert,
                        'adjusted_confidence': adjusted_confidence
                    })
            
            results.append(sequence_results)
        
        return results


if __name__ == "__main__":
    # Test the enhanced model
    batch_size = 2
    seq_len = 50
    signal_length = 100
    num_classes = 3  # Health + 2 defect types
    
    # Create random input
    x = torch.randn(batch_size, seq_len, signal_length)
    
    # Create random targets
    targets = []
    for b in range(batch_size):
        sequence_targets = []
        for i in range(seq_len):
            label = torch.randint(0, num_classes, (1,)).item()
            bbox = torch.rand(4) if label > 0 else torch.zeros(4)
            sequence_targets.append({
                'label': label,
                'bbox': bbox
            })
        targets.append(sequence_targets)
    
    # Create model
    model = EnhancedSignalSequenceDetector(
        signal_length=signal_length,
        d_model=256,
        num_classes=num_classes
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    loss, loss_dict = model(x, targets)
    
    print(f"Model output - Loss: {loss.item()}")
    print(f"Loss components: {loss_dict}")
    
    # Test prediction
    preds = model.predict(x)
    print(f"Predictions for first sequence: {preds[0]}")
