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
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleSignalEncoder(nn.Module):
    """
    Multi-scale signal encoder with 1D convolutions.
    """
    def __init__(self, signal_length, d_model=128, dropout=0.1):
        super(MultiScaleSignalEncoder, self).__init__()
        
        # Multi-scale convolutions
        self.conv_small = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU()
        )
        
        self.conv_medium = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU()
        )
        
        self.conv_large = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU()
        )
        
        self.conv_xlarge = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU()
        )
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, signal_length = x.shape
        
        # Reshape for 1D convolutions
        x = x.view(batch_size * seq_len, 1, signal_length)
        
        # Apply multi-scale convolutions
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        x_xlarge = self.conv_xlarge(x)
        
        # Concatenate multi-scale features
        x_concat = torch.cat([x_small, x_medium, x_large, x_xlarge], dim=1)
        
        # Apply adaptive pooling (reduces to single value per channel)
        x_pooled = self.pool(x_concat).squeeze(-1)
        
        # Reshape to [batch_size, seq_len, d_model]
        x_reshaped = x_pooled.view(batch_size, seq_len, -1)
        
        # Final projection
        x_projected = self.projection(x_reshaped)
        
        return x_projected


class SequenceTransformer(nn.Module):
    """
    Transformer for processing sequences of signal features.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(SequenceTransformer, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask (optional, for padding)
        # mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Apply transformer encoder
        # output = self.transformer_encoder(x, mask)
        output = self.transformer_encoder(x)
        
        # Apply layer normalization
        output = self.norm(output)
        
        # Extract attention weights (for visualization)
        # This is a simplified approach; in practice, you might need to modify the transformer code
        attention_weights = None
        
        return output, attention_weights


class DefectClassifier(nn.Module):
    """
    Classifier for detecting defects in signals.
    """
    def __init__(self, d_model=128, hidden_dim=64):
        super(DefectClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary classification: [no defect, defect]
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tuple of (logits, uncertainty), each with shape [batch_size, seq_len, 2]
        """
        logits = self.classifier(x)
        uncertainty = self.uncertainty(x) + 1e-6  # Add small constant for numerical stability
        
        return logits, uncertainty


class DefectPositionPredictor(nn.Module):
    """
    Predictor for defect positions within signals.
    """
    def __init__(self, d_model=128, hidden_dim=64):
        super(DefectPositionPredictor, self).__init__()
        
        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # [start, end] positions
            nn.Sigmoid()  # Normalize positions to [0, 1]
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tuple of (positions, uncertainty), each with shape [batch_size, seq_len, 2]
        """
        positions = self.position_predictor(x)
        uncertainty = self.uncertainty(x) + 1e-6  # Add small constant for numerical stability
        
        return positions, uncertainty


class TwoStageDefectDetector(nn.Module):
    """
    Two-stage model for defect detection and position prediction.
    """
    def __init__(self, signal_length, d_model=128, num_classes=2):
        super(TwoStageDefectDetector, self).__init__()
        
        # Signal encoder
        self.signal_encoder = MultiScaleSignalEncoder(signal_length, d_model=d_model)
        
        # Sequence transformer
        self.sequence_transformer = SequenceTransformer(d_model=d_model)
        
        # First stage: Defect classification
        self.defect_classifier = DefectClassifier(d_model=d_model)
        
        # Second stage: Position prediction
        self.position_predictor = DefectPositionPredictor(d_model=d_model)
        
    def forward(self, x, targets=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
            targets: Optional targets for training
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
        
        # 3. First stage: Defect classification
        defect_logits, defect_uncertainty = self.defect_classifier(sequence_features)
        defect_probs = F.softmax(defect_logits, dim=-1)
        
        # 4. Second stage: Position prediction
        position_preds, position_uncertainty = self.position_predictor(sequence_features)
        
        # 5. Weight position predictions by defect probability
        defect_weight = defect_probs[:, :, 1:2]  # Probability of being defective [batch_size, seq_len, 1]
        weighted_position_preds = position_preds * defect_weight.expand_as(position_preds)
        
        # 6. Return predictions or compute loss
        if targets is None:
            return {
                'defect_logits': defect_logits,
                'defect_probs': defect_probs,
                'defect_uncertainty': defect_uncertainty,
                'position_preds': weighted_position_preds,
                'position_uncertainty': position_uncertainty,
                'attention_weights': attention_weights
            }
        
        # 7. Compute loss
        loss_dict = self._compute_loss(
            defect_logits, defect_uncertainty,
            position_preds, position_uncertainty,
            defect_weight, targets
        )
        
        total_loss = (
            loss_dict['cls_loss'] + 
            loss_dict['position_loss'] + 
            0.05 * loss_dict['uncertainty_loss']
        )
        
        return total_loss, loss_dict
    
    def _compute_loss(self, defect_logits, defect_uncertainty, 
                     position_preds, position_uncertainty, 
                     defect_weight, targets):
        """
        Compute loss for training.
        
        Args:
            defect_logits: Tensor, shape [batch_size, seq_len, 2]
            defect_uncertainty: Tensor, shape [batch_size, seq_len, 2]
            position_preds: Tensor, shape [batch_size, seq_len, 2]
            position_uncertainty: Tensor, shape [batch_size, seq_len, 2]
            defect_weight: Tensor, shape [batch_size, seq_len, 1]
            targets: Dictionary with targets information
        
        Returns:
            dict with loss components
        """
        batch_size, seq_len, _ = defect_logits.shape
        device = defect_logits.device
        
        # Prepare target tensors
        target_classes = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        target_positions = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=device)
        
        # Process targets
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
        
        # Convert target classes to binary (0: no defect, 1: defect)
        binary_targets = (target_classes > 0).long()
        
        # Classification loss (binary cross-entropy)
        cls_loss = F.cross_entropy(
            defect_logits.view(-1, 2),
            binary_targets.view(-1)
        )
        
        # Position prediction loss (only for defective signals)
        defect_mask = (binary_targets == 1).unsqueeze(-1).expand_as(target_positions)
        
        if defect_mask.sum() > 0:
            # L1 loss for defect positions, weighted by defect probability
            pos_loss = F.l1_loss(
                position_preds[defect_mask],
                target_positions[defect_mask],
                reduction='mean'
            )
        else:
            pos_loss = torch.tensor(0.0, device=device)
        
        # Uncertainty regularization
        uncertainty_reg = (
            torch.mean(torch.exp(-defect_uncertainty) + defect_uncertainty) +
            torch.mean(torch.exp(-position_uncertainty) + position_uncertainty)
        ) * 0.01
        
        return {
            'cls_loss': cls_loss,
            'position_loss': pos_loss,
            'uncertainty_loss': uncertainty_reg,
            'total_loss': cls_loss + pos_loss + 0.05 * uncertainty_reg
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
            threshold: Float, threshold for defect prediction
            
        Returns:
            List of dicts with predictions for each sequence
        """
        # Get raw predictions
        with torch.no_grad():
            preds = self(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        defect_probs = preds['defect_probs']
        defect_uncertainty = preds['defect_uncertainty']
        position_preds = preds['position_preds']
        position_uncertainty = preds['position_uncertainty']
        
        results = []
        
        for b in range(batch_size):
            sequence_results = []
            
            for i in range(seq_len):
                # Get defect probability
                defect_prob = defect_probs[b, i, 1].item()
                defect_uncert = defect_uncertainty[b, i, 1].item()
                
                # Adjust confidence based on uncertainty
                adjusted_confidence = defect_prob / (1.0 + defect_uncert)
                
                # Add to results if confidence is above threshold
                if adjusted_confidence > threshold:
                    sequence_results.append({
                        'position': i,  # Position in sequence
                        'defect_prob': defect_prob,
                        'defect_uncertainty': defect_uncert,
                        'defect_position': position_preds[b, i].cpu().numpy(),  # Start and end positions within the signal
                        'position_uncertainty': position_uncertainty[b, i].cpu().numpy(),
                        'adjusted_confidence': adjusted_confidence
                    })
            
            results.append(sequence_results)
        
        return results
