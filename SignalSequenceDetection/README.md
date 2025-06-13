# Signal Sequence Defect Detection

This project implements a deep learning approach for detecting defects in sequences of signals. Instead of using image-based detection, this approach directly processes signal sequences and leverages temporal relationships between signals to improve defect detection accuracy.

## Key Features

- Processes raw signal data instead of converting to images
- Creates sequences of signals for context-aware detection
- Uses a transformer-based architecture to capture temporal relationships
- Implements an anomaly detection approach to identify defects against "health" patterns
- Provides tools for dataset preparation, training, and prediction

## Project Structure

- `dataset_preparation.py`: Prepares signal sequences from JSON files
- `model.py`: Implements the signal sequence detector model
- `train.py`: Training script with evaluation metrics
- `predict.py`: Script for making predictions on new data
- `README.md`: Project documentation

## Model Architecture

The model consists of several key components:

1. **Signal Encoder**: Extracts features from individual signals using 1D CNNs
2. **Sequence Transformer**: Processes sequences of signal features to capture temporal relationships
3. **Context Aggregator**: Aggregates context information across the sequence
4. **Anomaly Detector**: Detects anomalies by comparing signal features to sequence context
5. **Detection Head**: Predicts defect class and position

## Usage

### Dataset Preparation

```python
from dataset_preparation import SignalSequencePreparation

# Create signal sequences
prep = SignalSequencePreparation(
    ds_folder="WOT-20250522(auto)",
    output_folder="signal_dataset",
    seq_length=50
)
sequences, annotations = prep.create_signal_sequences()

# Visualize a sequence
prep.visualize_sequence(0, save_path="sequence_example.png")
```

### Training

```bash
python train.py
```

This will:
1. Load the prepared dataset
2. Create and train the model
3. Save checkpoints and training history
4. Evaluate the model on a test set

### Prediction

```bash
python predict.py --checkpoint signal_dataset/checkpoints/best_model.pth --data signal_dataset/signal_sequences.pt --output predictions
```

## Implementation Details

### Signal Sequence Creation

The dataset preparation process:
1. Loads JSON files containing signal data
2. Extracts signals and annotations for each beam
3. Creates sequences of signals with corresponding defect annotations
4. Normalizes signals and annotations for model input

### Model Training

The training process:
1. Splits the dataset into training, validation, and test sets
2. Trains the model with different learning rates for different components
3. Uses a cosine annealing learning rate scheduler
4. Saves checkpoints and training history
5. Evaluates the model on the test set

### Loss Function

The model uses a combination of losses:
1. Classification loss (cross-entropy)
2. Bounding box regression loss (L1)
3. Anomaly consistency loss (MSE)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm

## Future Improvements

- Implement data augmentation for signals
- Add support for different signal lengths
- Explore different transformer architectures
- Implement ensemble methods for improved accuracy
- Add support for real-time prediction
