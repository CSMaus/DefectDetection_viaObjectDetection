import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from dataset_preparation import SignalSequenceDataset
from model import SignalSequenceDetector


def load_model(checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        model: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters from checkpoint
    if 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
    else:
        # Default parameters if not found in checkpoint
        model_params = {
            'signal_length': 100,
            'd_model': 128,
            'num_classes': 2,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1
        }
    
    # Create model
    model = SignalSequenceDetector(**model_params).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def predict_sequence(model, signals, threshold=0.5, device='cuda'):
    """
    Make predictions for a sequence of signals.
    
    Args:
        model: Model to use for prediction
        signals: Tensor of signals, shape [1, seq_len, signal_length]
        threshold: Confidence threshold for predictions
        device: Device to run prediction on
        
    Returns:
        list: Predictions for the sequence
    """
    model.eval()
    
    with torch.no_grad():
        # Move signals to device
        signals = signals.to(device)
        
        # Get predictions
        preds = model.predict(signals, threshold=threshold)
    
    return preds[0]  # Return predictions for the first (and only) sequence


def visualize_predictions(signals, predictions, save_path=None, max_signals=10):
    """
    Visualize predictions for a sequence of signals.
    
    Args:
        signals: Tensor of signals, shape [seq_len, signal_length]
        predictions: List of predictions
        save_path: Path to save the visualization
        max_signals: Maximum number of signals to visualize
    """
    # Convert to numpy if tensor
    if torch.is_tensor(signals):
        signals = signals.cpu().numpy()
    
    # Limit number of signals to visualize
    num_signals = min(signals.shape[0], max_signals)
    
    # Create figure
    fig, axs = plt.subplots(num_signals, 1, figsize=(15, 2 * num_signals))
    
    # If only one signal, wrap axs in a list
    if num_signals == 1:
        axs = [axs]
    
    # Plot each signal
    for i in range(num_signals):
        axs[i].plot(signals[i])
        axs[i].set_title(f"Signal {i}")
        axs[i].grid(True)
        
        # Add predictions if available
        for pred in predictions:
            if pred['position'] == i:
                # Get defect position within the signal
                start, end = pred['defect_position']
                start_idx = int(start * len(signals[i]))
                end_idx = int(end * len(signals[i]))
                
                # Highlight prediction
                axs[i].axvspan(start_idx, end_idx, alpha=0.3, color='red')
                
                # Add label
                axs[i].text(
                    start_idx,
                    max(signals[i]),
                    f"Class {pred['class']} ({pred['class_score']:.2f})",
                    fontsize=8
                )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def predict_dataset(model, dataset, output_dir, device='cuda', threshold=0.5, batch_size=8):
    """
    Make predictions for a dataset and save results.
    
    Args:
        model: Model to use for prediction
        dataset: Dataset to predict on
        output_dir: Directory to save results
        device: Device to run prediction on
        threshold: Confidence threshold for predictions
        batch_size: Batch size for prediction
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize results
    all_predictions = []
    
    # Make predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
            # Get data
            signals = batch['signals'].to(device)
            file_names = batch['file_name']
            scan_keys = batch['scan_key']
            
            # Get predictions
            batch_preds = model.predict(signals, threshold=threshold)
            
            # Store predictions
            for i, preds in enumerate(batch_preds):
                all_predictions.append({
                    'file_name': file_names[i],
                    'scan_key': scan_keys[i],
                    'predictions': preds
                })
            
            # Visualize first batch
            if batch_idx == 0:
                for i in range(min(5, len(batch_preds))):
                    visualize_predictions(
                        signals[i],
                        batch_preds[i],
                        save_path=os.path.join(output_dir, f'sample_prediction_{i}.png'),
                        max_signals=10
                    )
    
    # Save all predictions
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"Predictions saved to {output_dir}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Predict defects in signal sequences')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    
    # Load dataset
    print(f"Loading dataset from {args.data}")
    dataset = SignalSequenceDataset(args.data)
    print(f"Dataset size: {len(dataset)}")
    
    # Make predictions
    print(f"Making predictions with threshold {args.threshold}")
    predict_dataset(
        model=model,
        dataset=dataset,
        output_dir=args.output,
        device=device,
        threshold=args.threshold,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
