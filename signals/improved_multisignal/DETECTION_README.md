# Detection Model Training and Testing

This directory contains enhanced scripts for training and testing detection models with comprehensive checkpoint management and evaluation.

## Files Overview

### Training Scripts
- **`train_detection.py`** - Enhanced training script with checkpoint saving and history tracking
- **`enhanced_position_training.py`** - Your existing enhanced position model training

### Testing Scripts
- **`test_detection_model.py`** - Comprehensive testing with detailed evaluation plots and metrics
- **`simple_test_detection.py`** - Quick testing script for basic metrics

### Model Files
- **`detection_models/simple_detection_model.py`** - Simple detection model with enhanced transformer
- **`detection_models/complex_detection_model.py`** - Complex detection model with multi-scale convolutions

## Enhanced Training Features

The updated `train_detection.py` script now includes:

1. **Checkpoint Management**
   - Saves model checkpoint every epoch
   - Checkpoints saved in `models/{ModelName}_{timestamp}/`
   - Each checkpoint includes model state, optimizer state, scheduler state, and training history

2. **Training History Tracking**
   - Tracks loss, accuracy, and learning rate for each epoch
   - Saves training history as JSON file
   - Creates comprehensive training plots automatically

3. **Automatic Directory Structure**
   ```
   models/
   ├── Simple_20250717_0800/
   │   ├── epoch_01_checkpoint.pth
   │   ├── epoch_02_checkpoint.pth
   │   ├── ...
   │   ├── best_simple_detection.pth
   │   ├── training_history.json
   │   └── training_history.png
   └── Complex_20250717_0800/
       ├── epoch_01_checkpoint.pth
       ├── ...
       └── best_complex_detection.pth
   ```

## Usage

### Training Models
```bash
python train_detection.py
```

This will:
- Train both Simple and Complex detection models
- Save checkpoints every epoch with timestamp
- Create training history plots
- Save best models based on validation accuracy
- Generate comprehensive results summary

### Testing Models

#### Comprehensive Testing
```bash
python test_detection_model.py
```

Features:
- Automatically finds and loads best model checkpoints
- Creates detailed evaluation plots (ROC curves, confusion matrices, etc.)
- Calculates comprehensive metrics (precision, recall, F1, AUC)
- Saves results in timestamped directory

#### Quick Testing
```bash
python simple_test_detection.py
```

Features:
- Quick evaluation with basic metrics
- Minimal output for fast checking
- Useful for quick model comparison

## Model Architecture

### Simple Detection Model
- Minimal preprocessing with direct linear projection
- Enhanced transformer (8 layers, 16 heads, 128 d_model)
- Designed to let transformer learn signal patterns directly

### Complex Detection Model
- Multi-scale 1D convolution preprocessing
- Standard transformer (4 layers, 8 heads, 64 d_model)
- Extracts features at different scales before transformer processing

## Training Configuration

Current settings in `train_detection.py`:
- **Epochs**: 15
- **Batch Size**: 16
- **Sequence Length**: 50
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Data**: json_data_0717 with 20% validation split

## Output Structure

### Training Outputs
- Model checkpoints with full training state
- Training history JSON with all metrics
- Training plots showing loss, accuracy, and learning rate curves
- Best model selection based on validation accuracy

### Testing Outputs
- Comprehensive evaluation plots
- Detailed metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrices and ROC curves
- Probability distribution analysis

## Checkpoint Format

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'train_loss': float,
    'val_loss': float,
    'train_accuracy': float,
    'val_accuracy': float,
    'history': dict  # Complete training history
}
```

## Tips

1. **Monitor Training**: Check the generated training plots to identify overfitting or convergence issues
2. **Model Selection**: The script automatically saves the best model based on validation accuracy
3. **Checkpoint Recovery**: You can resume training from any checkpoint if needed
4. **Comparison**: Use the comprehensive testing script to compare different models
5. **Quick Checks**: Use the simple testing script during development for fast feedback

## Troubleshooting

- **CUDA Memory**: Reduce batch size if you encounter GPU memory issues
- **Missing Checkpoints**: Check the models directory structure and timestamps
- **Data Loading**: Ensure json_data_0717 directory exists and contains proper data files
