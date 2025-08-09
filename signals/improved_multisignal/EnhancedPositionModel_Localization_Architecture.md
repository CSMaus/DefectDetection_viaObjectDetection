# EnhancedPositionMultiSignalClassifier - Localization Architecture

## Overview
**Purpose**: Defect position localization (finding where defects are located within signals)  
**Input**: Multi-signal sequences `(batch_size, num_signals, signal_length)`  
**Output**: Defect start and end positions `(batch_size, num_signals)` each  
**Note**: Although this model has detection capability, you use it ONLY for localization

## Localization-Specific Architecture

### 1. Enhanced Feature Extraction for Position Detection
```python
self.conv1d = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```
**Localization Function**: Extracts fine-grained features needed for precise position detection
- **Smaller Channel Count**: 16→32 (vs 32→64 in detection model) for position-focused features
- **Kernel Size 3**: Maintains high spatial resolution for accurate localization

### 2. Advanced Background Trend Removal
```python
self.background_extractor = nn.Sequential(
    nn.Conv1d(32, 32, kernel_size=15, padding=7, groups=32),   # Medium-range trends
    nn.BatchNorm1d(32),
    nn.Conv1d(32, 32, kernel_size=31, padding=15, groups=32),  # Long-range trends
)
```
**Localization Function**: Critical for position accuracy
- **Dual-Scale Extraction**: Removes both medium (15) and long-range (31) background trends
- **Groups=32**: Depth-wise convolution preserves spatial information
- **Background Subtraction**: `x = x - bg_trend` isolates defect-specific patterns

### 3. Position-Aware Feature Processing
```python
self.shared_layer = nn.Sequential(
    nn.Linear(signal_length, hidden_sizes[0]),  # Preserves spatial information
    nn.Dropout(dropout),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.Dropout(dropout),
    nn.ReLU(),
)
```
**Localization Function**: Maintains spatial resolution throughout processing

### 4. Transformer for Spatial Context
```python
self.transformer_layers = nn.ModuleList([
    TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
    for _ in range(num_transformer_layers)
])
```
**Localization Function**: Captures spatial relationships between signal regions
- **Self-Attention**: Models long-range spatial dependencies
- **Local Attention**: Focuses on neighboring regions (kernel=9)
- **Multi-Layer**: 4 layers for complex spatial pattern recognition

## Dual Position Prediction System

### 5. Fine-Grained Position Head (Primary)
```python
self.position_head = nn.Sequential(
    # Layer 1: Extract position-relevant features
    nn.Linear(hidden_sizes[1], hidden_sizes[1]),
    nn.LayerNorm(hidden_sizes[1]),
    nn.ReLU(),
    nn.Dropout(dropout * 0.5),
    
    # Layer 2: Refine position features  
    nn.Linear(hidden_sizes[1], hidden_sizes[1]),
    nn.LayerNorm(hidden_sizes[1]),
    nn.ReLU(),
    nn.Dropout(dropout * 0.5),
    
    # Layer 3: Position-specific processing
    nn.Linear(hidden_sizes[1], hidden_sizes[1] // 2),
    nn.LayerNorm(hidden_sizes[1] // 2),
    nn.ReLU(),
    nn.Dropout(dropout * 0.3),
    
    # Final: Output [start, end] positions
    nn.Linear(hidden_sizes[1] // 2, 2)
)
```
**Localization Function**: High-precision position regression
- **Deep Architecture**: 4 layers for complex position mapping
- **Layer Normalization**: Stabilizes position gradient flow
- **Reduced Dropout**: Lower dropout in later layers for precision
- **Output**: Raw [start, end] coordinates

### 6. Coarse-Grained Position Head (Robustness)
```python
self.position_head_coarse = nn.Sequential(
    nn.Linear(hidden_sizes[1], hidden_sizes[1] // 4),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1] // 4, 2)  # Simple coarse prediction
)
```
**Localization Function**: Provides robust backup predictions
- **Shallow Architecture**: Simple mapping for general position estimates
- **Smaller Hidden Size**: Focuses on broad position patterns

## Localization Data Flow

### Position Prediction Process:
1. **Input**: `(batch_size, num_signals, signal_length)` raw signals
2. **Feature Extraction**: Conv1d → `(batch * num_signals, 32, signal_length)`
3. **Background Removal**: Extract and subtract trends
4. **Spatial Pooling**: Global average → `(batch * num_signals, signal_length)`
5. **Feature Processing**: Shared layers → `(batch * num_signals, hidden_size)`
6. **Reshape**: `(batch_size, num_signals, hidden_size)` for sequence processing
7. **Position Encoding**: Add spatial position information
8. **Transformer**: 4 layers → spatial feature refinement
9. **Dual Prediction**:
   - **Fine**: `position_head(features)` → precise positions
   - **Coarse**: `position_head_coarse(features)` → robust positions
10. **Combination**: `0.7 * fine + 0.3 * coarse` → balanced prediction
11. **Constraints**: Apply position validation and normalization

## Position Constraint System

### Position Validation and Normalization:
```python
# Normalize to [0,1] range
defect_start = torch.sigmoid(position_combined[:, :, 0])
defect_end = torch.sigmoid(position_combined[:, :, 1])

# Ensure start <= end
min_pos = torch.minimum(defect_start, defect_end)
max_pos = torch.maximum(defect_start, defect_end)
defect_start = min_pos
defect_end = max_pos

# Prevent zero-length defects
gap = 0.01
defect_end = torch.maximum(defect_end, defect_start + gap)
defect_end = torch.clamp(defect_end, max=1.0)
```

**Localization Function**: Ensures valid position outputs
- **Sigmoid Normalization**: Maps raw outputs to [0,1] range
- **Order Constraint**: Guarantees start ≤ end
- **Minimum Length**: Prevents degenerate zero-length predictions
- **Boundary Clipping**: Keeps positions within valid signal range

## Key Localization Features

### Multi-Scale Position Prediction:
- **Fine Head (70%)**: High-precision position regression
- **Coarse Head (30%)**: Robust general position estimation
- **Weighted Combination**: Balances precision and robustness

### Spatial Processing Optimizations:
- **Background Trend Removal**: Critical for accurate position detection
- **Preserved Spatial Resolution**: Maintains fine-grained spatial information
- **Transformer Spatial Attention**: Models complex spatial relationships
- **Position Constraints**: Ensures physically valid outputs

### Localization-Specific Design:
- **Dual-Scale Background Removal**: Handles various trend patterns
- **Deep Position Heads**: Complex position mapping capability
- **Spatial Context Modeling**: Transformer captures position relationships
- **Robust Constraint System**: Prevents invalid position predictions

## Usage for Localization Only

### Input Requirements:
- **Preprocessed Signals**: Normalized signal sequences
- **Signal Length**: Variable (handled by adaptive processing)
- **Batch Processing**: Supports multiple signals simultaneously

### Output Interpretation:
- **defect_start**: Normalized start position (0.0 to 1.0)
- **defect_end**: Normalized end position (0.0 to 1.0)
- **Position Range**: Multiply by actual signal length for absolute positions
- **Minimum Gap**: 0.01 normalized units between start and end

### C# Integration:
- **ONNX Compatible**: Exported for C# runtime inference
- **Input Format**: Float arrays matching training format
- **Output Format**: Two float arrays (start_positions, end_positions)
- **Post-Processing**: Apply position constraints and denormalization
