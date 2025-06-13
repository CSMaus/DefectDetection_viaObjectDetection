import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
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
        return x + self.pe[:, :x.size(1), :]


class SignalEncoder(nn.Module):
    """
    Encoder for individual signals.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128):
        super(SignalEncoder, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim*4)
        
        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim*4, output_dim)
        
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
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Final projection
        x = self.fc(x)
        
        # Reshape back to sequence
        x = x.view(batch_size, seq_len, -1)
        
        return x


class SequenceTransformer(nn.Module):
    """
    Transformer for processing sequences of signal embeddings.
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(SequenceTransformer, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [seq_len, seq_len]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return x


class DefectDetectionHead(nn.Module):
    """
    Detection head for predicting defect class and position within 1D signals.
    """
    def __init__(self, d_model=128, num_classes=2):
        super(DefectDetectionHead, self).__init__()
        
        # Class prediction
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Defect position prediction (start, end) within the 1D signal
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid()  # Normalize to [0, 1] range of the signal length
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            class_logits: Tensor, shape [batch_size, seq_len, num_classes]
            position_pred: Tensor, shape [batch_size, seq_len, 2] (start, end positions in signal)
        """
        class_logits = self.class_head(x)
        position_pred = self.position_head(x)
        
        return class_logits, position_pred


class ContextAggregator(nn.Module):
    """
    Aggregates context information across the sequence.
    """
    def __init__(self, d_model=128):
        super(ContextAggregator, self).__init__()
        
        # Bidirectional GRU for context aggregation
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Projection layer
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        # Apply bidirectional GRU
        context, _ = self.gru(x)
        
        # Project back to original dimension
        context = self.projection(context)
        
        return context


class AnomalyDetector(nn.Module):
    """
    Detects anomalies by comparing signal features to sequence context.
    """
    def __init__(self, d_model=128, hidden_dim=64):
        super(AnomalyDetector, self).__init__()
        
        # Anomaly detection network
        self.anomaly_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, signal_features, context_features):
        """
        Args:
            signal_features: Tensor, shape [batch_size, seq_len, d_model]
            context_features: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, 1]
        """
        # Concatenate signal and context features
        combined = torch.cat([signal_features, context_features], dim=-1)
        
        # Compute anomaly scores
        anomaly_scores = self.anomaly_net(combined)
        
        return anomaly_scores


class SignalSequenceDetector(nn.Module):
    """
    Complete model for detecting defects in sequences of signals.
    """
    def __init__(
        self,
        signal_length=100,
        d_model=128,
        num_classes=2,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):
        super(SignalSequenceDetector, self).__init__()
        
        # Signal encoder
        self.signal_encoder = SignalEncoder(
            input_dim=1,
            hidden_dim=64,
            output_dim=d_model
        )
        
        # Sequence transformer
        self.sequence_transformer = SequenceTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Context aggregator
        self.context_aggregator = ContextAggregator(d_model=d_model)
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(d_model=d_model, hidden_dim=64)
        
        # Detection head
        self.detection_head = DefectDetectionHead(d_model=d_model, num_classes=num_classes)
        
        # Health feature extractor (to learn common "health" patterns)
        self.health_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Attention mechanism for focusing on important signals
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, x, targets=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, signal_length]
            targets: List of dicts with 'label' and position information
        Returns:
            If targets is None:
                dict with predictions
            Else:
                loss, dict with loss components
        """
        batch_size, seq_len, signal_length = x.shape
        
        # 1. Encode individual signals
        signal_features = self.signal_encoder(x)
        
        # 2. Apply sequence transformer to capture temporal relationships
        sequence_features = self.sequence_transformer(signal_features)
        
        # 3. Aggregate context information
        context_features = self.context_aggregator(sequence_features)
        
        # 4. Extract health features (common patterns)
        health_features = self.health_extractor(sequence_features)
        
        # 5. Compute attention weights
        attention_weights = self.attention(sequence_features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 6. Apply attention to enhance important signals
        enhanced_features = sequence_features * attention_weights + context_features
        
        # 7. Detect anomalies by comparing to health features
        anomaly_scores = self.anomaly_detector(enhanced_features, health_features)
        
        # 8. Predict class and defect positions within signals
        class_logits, position_pred = self.detection_head(enhanced_features)
        
        # 9. Enhance class predictions with anomaly scores
        # For non-health classes (index > 0), add anomaly score
        if class_logits.size(-1) > 1:  # More than just health/non-health
            # Clone to avoid in-place modification
            enhanced_logits = class_logits.clone()
            
            # Add anomaly scores to non-health classes
            enhanced_logits[:, :, 1:] = enhanced_logits[:, :, 1:] + anomaly_scores
            
            class_logits = enhanced_logits
        
        # 10. Return predictions or compute loss
        if targets is None:
            return {
                'class_preds': class_logits,
                'position_preds': position_pred,  # Start and end positions within signals
                'anomaly_scores': anomaly_scores,
                'attention_weights': attention_weights
            }
        
        # 11. Compute loss
        loss_dict = self._compute_loss(class_logits, position_pred, anomaly_scores, targets)
        total_loss = loss_dict['cls_loss'] + loss_dict['position_loss'] + 0.1 * loss_dict['anomaly_consistency_loss']
        
        return total_loss, loss_dict
    
    def _compute_loss(self, class_logits, position_pred, anomaly_scores, targets):
        """
        Compute loss for training.
        
        Args:
            class_logits: Tensor, shape [batch_size, seq_len, num_classes]
            position_pred: Tensor, shape [batch_size, seq_len, 2] (start, end positions in signal)
            anomaly_scores: Tensor, shape [batch_size, seq_len, 1]
            targets: List of dicts with 'label' and position information
        
        Returns:
            dict with loss components
        """
        batch_size, seq_len, num_classes = class_logits.shape
        device = class_logits.device
        
        # Prepare target tensors
        target_classes = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        target_positions = torch.zeros((batch_size, seq_len, 2), dtype=torch.float32, device=device)
        
        # Fill target tensors
        for b in range(batch_size):
            for i, target in enumerate(targets[b]):
                if i < seq_len:
                    target_classes[b, i] = target['label']
                    
                    # Extract defect start and end positions within the signal
                    if target['label'] > 0:  # If it's a defect (not "Health")
                        if torch.is_tensor(target['bbox']):
                            if target['bbox'].numel() >= 4:
                                # The defect positions are the last two values in bbox
                                target_positions[b, i, 0] = target['bbox'][2]  # defect start position
                                target_positions[b, i, 1] = target['bbox'][3]  # defect end position
                        else:
                            # Handle case where bbox might be a list or numpy array
                            target_positions[b, i, 0] = target['bbox'][2]  # defect start position
                            target_positions[b, i, 1] = target['bbox'][3]  # defect end position
        
        # Classification loss
        cls_loss = F.cross_entropy(
            class_logits.view(-1, num_classes),
            target_classes.view(-1)
        )
        
        # Position prediction loss (only for defect signals)
        positive_mask = (target_classes > 0).unsqueeze(-1).expand_as(target_positions)
        
        if positive_mask.sum() > 0:
            # L1 loss for defect positions
            position_loss = F.l1_loss(
                position_pred[positive_mask],
                target_positions[positive_mask]
            )
        else:
            position_loss = torch.tensor(0.0, device=device)
        
        # Anomaly consistency loss
        anomaly_consistency_loss = 0.0
        if seq_len > 1:
            # Calculate temporal consistency loss for anomaly scores
            for t in range(1, seq_len):
                anomaly_consistency_loss += F.mse_loss(
                    anomaly_scores[:, t], anomaly_scores[:, t-1]
                )
            anomaly_consistency_loss /= (seq_len - 1)
        
        return {
            'cls_loss': cls_loss,
            'position_loss': position_loss,
            'anomaly_consistency_loss': anomaly_consistency_loss,
            'total_loss': cls_loss + position_loss + 0.1 * anomaly_consistency_loss
        }
    
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
        position_preds = preds['position_preds']
        anomaly_scores = preds['anomaly_scores']
        
        results = []
        
        for b in range(batch_size):
            sequence_results = []
            
            for i in range(seq_len):
                # Get class probabilities
                class_probs = F.softmax(class_preds[b, i], dim=0)
                
                # Get predicted class (argmax)
                pred_class = torch.argmax(class_probs).item()
                class_score = class_probs[pred_class].item()
                
                # Get defect position within the signal
                pred_position = position_preds[b, i].cpu().numpy()
                
                # Get anomaly score
                anomaly_score = anomaly_scores[b, i, 0].item()
                
                # Add to results if confidence is above threshold
                if class_score > threshold or (pred_class > 0 and anomaly_score > threshold):
                    sequence_results.append({
                        'position': i,  # Position in sequence
                        'class': pred_class,
                        'class_score': class_score,
                        'defect_position': pred_position,  # Start and end positions within the signal
                        'anomaly_score': anomaly_score
                    })
            
            results.append(sequence_results)
        
        return results


if __name__ == "__main__":
    # Test the model
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
    model = SignalSequenceDetector(
        signal_length=signal_length,
        d_model=128,
        num_classes=num_classes
    )
    
    # Forward pass
    loss, loss_dict = model(x, targets)
    
    print(f"Model output - Loss: {loss.item()}")
    print(f"Loss components: {loss_dict}")
    
    # Test prediction
    preds = model.predict(x)
    print(f"Predictions for first sequence: {preds[0]}")
