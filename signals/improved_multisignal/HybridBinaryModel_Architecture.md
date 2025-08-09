# HybridBinaryModel Architecture Documentation

## Overview
**Purpose**: Binary classification only (defect/no-defect detection)  
**Input**: Multi-signal sequences `(batch_size, num_signals, signal_length=320)`  
**Output**: Defect probability per signal `(batch_size, num_signals)`  

## Architecture Components

### 1. Feature Extraction Layer (from direct_defect.py)
```python
self.conv_layers = nn.Sequential(
    nn.Conv1d(1, 32, kernel_size=3, padding=1),     # Extract local patterns
    nn.BatchNorm1d(32),
    nn.ReLU(),
    
    nn.Conv1d(32, 64, kernel_size=3, padding=1),    # Deeper feature extraction
    nn.BatchNorm1d(64),
    nn.ReLU(),
    
    nn.Conv1d(64, 64, kernel_size=5, padding=2),    # Wider context (kernel=5)
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```
**Function**: Extracts hierarchical features from raw 1D signals
- **Layer 1**: Basic pattern detection (32 filters)
- **Layer 2**: Complex pattern recognition (64 filters)  
- **Layer 3**: Context-aware features (kernel size 5 for wider receptive field)

### 2. ONNX-Compatible Pooling
```python
kernel_size = signal_length // 128  # Adaptive kernel size
self.fixed_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)
```
**Function**: Reduces signal length to exactly 128 features for consistent processing
- **Adaptive**: Kernel size adjusts based on input signal length
- **ONNX-Compatible**: Uses fixed pooling operations for export compatibility

### 3. Feature Normalization
```python
x = F.interpolate(x, size=128, mode='linear', align_corners=False)
x = x.mean(dim=1)  # Global average pooling -> (batch * num_signals, 128)
```
**Function**: Ensures consistent feature dimensionality across all inputs

### 4. Shared Feature Processing
```python
self.shared_layer = nn.Sequential(
    nn.Linear(128, hidden_sizes[0]),    # Default: 128 -> 128
    nn.Dropout(dropout),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # Default: 128 -> 64
    nn.Dropout(dropout),
    nn.ReLU(),
)
```
**Function**: Transforms raw features into higher-level representations

### 5. Positional Encoding
```python
class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))
```
**Function**: Adds position information to help model understand signal sequence context
- **Learnable**: Position encodings are trained parameters
- **Relative**: Focuses on relationships between signal positions

### 6. Transformer Architecture
```python
self.transformer_layers = nn.ModuleList([
    TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
    for _ in range(num_transformer_layers)  # Default: 4 layers
])
```

#### TransformerEncoder Components:
- **Multi-Head Self-Attention**: Captures long-range dependencies between signals
- **Local Attention**: Convolutional attention focusing on neighboring signals (kernel=9)
- **Feed-Forward Network**: Non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Prevents vanishing gradients

### 7. Binary Classification Head
```python
self.classifier = nn.Linear(hidden_sizes[1], 1)  # 64 -> 1
defect_prob = torch.sigmoid(defect_logits)
```
**Function**: Final binary classification
- **Output**: Single probability value per signal (0-1)
- **Activation**: Sigmoid for probability interpretation

## Data Flow

### Forward Pass Process:
1. **Input**: `(batch_size, num_signals, 320)` raw signals
2. **Reshape**: `(batch_size * num_signals, 1, 320)` for 1D convolution
3. **Feature Extraction**: Conv layers → `(batch_size * num_signals, 64, 320)`
4. **Pooling**: Adaptive pooling → `(batch_size * num_signals, 64, ~128)`
5. **Normalization**: Interpolate to exactly 128 → `(batch_size * num_signals, 64, 128)`
6. **Global Pooling**: Average pooling → `(batch_size * num_signals, 128)`
7. **Shared Processing**: Linear layers → `(batch_size * num_signals, 64)`
8. **Reshape**: `(batch_size, num_signals, 64)` for sequence processing
9. **Position Encoding**: Add positional information
10. **Transformer**: 4 layers of attention → `(batch_size, num_signals, 64)`
11. **Classification**: Linear + Sigmoid → `(batch_size, num_signals)` probabilities

## Key Features

### Hybrid Architecture Benefits:
- **CNN Feature Extraction**: Captures local signal patterns effectively
- **Transformer Processing**: Models relationships between multiple signals
- **ONNX Compatibility**: Fixed operations for easy deployment
- **Proven Performance**: Based on working detection model architecture

### Design Decisions:
- **Binary Focus**: Optimized specifically for defect/no-defect classification
- **Multi-Scale Features**: Combines local (CNN) and global (Transformer) patterns
- **Robust Pooling**: Handles variable input sizes consistently
- **Dropout Regularization**: Prevents overfitting during training

## Model Parameters
- **Default Hidden Sizes**: [128, 64, 32]
- **Transformer Heads**: 8
- **Transformer Layers**: 4
- **Dropout Rate**: 0.1
- **Signal Length**: 320 (fixed)
- **Output Features**: 128 (after pooling)

## Usage in C# Application
- **Input Format**: Normalized signal sequences
- **Output Format**: Probability values (0.0 to 1.0)
- **Threshold**: Typically 0.5 for binary decision
- **ONNX Export**: Compatible with ONNX runtime for inference
