# Improved MultiSignal Classifier

This is an improved version of the MultiSignalClassifier_N model with several enhancements:

## Key Improvements

1. **Increased Context Window**
   - Local attention kernel size increased from 5 to 9
   - Background extractor kernel size increased from 11 to 15
   - This allows the model to capture wider context without losing accuracy

2. **Deeper Feature Extraction**
   - Increased number of transformer layers from 1 to 4
   - Added dropout for better regularization
   - Enhanced batch normalization for more stable training

3. **Fixed Sequence Length**
   - Set to exactly 50 signals per sequence as requested
   - Position encoding max_len kept at 300 as in the original model

4. **Focus on Sequences with Defects**
   - Dataset now only includes sequences that contain at least one defect
   - This ensures more efficient training on relevant data

5. **JSON Data Support**
   - Added support for loading data from JSON files instead of folder structure
   - Maintains the same data structure as the original implementation

6. **Training Enhancements**
   - Added learning rate scheduling
   - Implemented early stopping
   - Added gradient clipping to prevent exploding gradients
   - Enhanced loss function to focus on defective signals

## Files

- `improved_model.py`: Contains the improved model architecture
- `json_dataset.py`: Dataset class for loading data from JSON files
- `training.py`: Script for training the model with validation and early stopping

## Usage

1. Update the `json_dir` variable in `training.py` to point to your JSON files
2. Run the training script:
   ```
   python training.py
   ```

3. The script will:
   - Train the model
   - Save checkpoints
   - Plot training history
   - Export the model to ONNX format

## Model Architecture

The ImprovedMultiSignalClassifier maintains the same basic structure as the original MultiSignalClassifier_N but with the following enhancements:

- Increased convolutional kernel sizes for wider context
- More transformer layers for deeper feature extraction
- Added dropout for regularization
- Enhanced batch normalization
- Fixed sequence length of 50

## Data Format

The JSON files should have the following structure:

```json
{
  "scan_key_1": {
    "signal_1_Health": {
      "signal": [0.1, 0.2, 0.3, ...]
    },
    "signal_2_Delamination_0.1-0.3": {
      "signal": [0.2, 0.3, 0.4, ...]
    },
    ...
  },
  "scan_key_2": {
    ...
  }
}
```

Where:
- `scan_key_X` corresponds to different scan sequences
- Signal keys follow the format: `{index}_{defect_type}_{start-end}` or `{index}_Health`
- The `signal` field contains the actual signal data as an array
