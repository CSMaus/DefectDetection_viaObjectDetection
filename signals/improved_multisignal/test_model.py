import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from improved_model import ImprovedMultiSignalClassifier
from json_dataset import JsonSignalDataset


def load_model(model_path, signal_length=320, hidden_sizes=[128, 64, 32], num_heads=8, num_transformer_layers=4):
    """
    Load a trained model from a checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        signal_length: Length of input signals
        hidden_sizes: Hidden layer sizes
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
        
    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ImprovedMultiSignalClassifier(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def visualize_predictions(model, signals, labels, defect_positions, threshold=0.5):
    """
    Visualize model predictions on a sequence of signals
    
    Args:
        model: Trained model
        signals: Signal tensor [num_signals, signal_length]
        labels: Ground truth labels [num_signals]
        defect_positions: Ground truth defect positions [num_signals, 2]
        threshold: Confidence threshold for defect detection
    """
    device = next(model.parameters()).device
    signals_tensor = signals.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get predictions
    with torch.no_grad():
        defect_prob, defect_start, defect_end = model(signals_tensor)
    
    # Convert to numpy
    defect_prob = defect_prob[0].cpu().numpy()
    defect_start = defect_start[0].cpu().numpy()
    defect_end = defect_end[0].cpu().numpy()
    signals_np = signals.cpu().numpy()
    labels_np = labels.cpu().numpy()
    defect_positions_np = defect_positions.cpu().numpy()
    
    # Create figure
    num_signals = min(10, len(signals_np))  # Show at most 10 signals
    fig, axes = plt.subplots(num_signals, 1, figsize=(15, 2*num_signals))
    
    if num_signals == 1:
        axes = [axes]
    
    for i in range(num_signals):
        ax = axes[i]
        signal = signals_np[i]
        
        # Plot signal
        ax.plot(signal, 'b-', alpha=0.7)
        
        # Plot ground truth
        if labels_np[i] > 0.5:
            start_idx = int(defect_positions_np[i, 0] * len(signal))
            end_idx = int(defect_positions_np[i, 1] * len(signal))
            ax.axvspan(start_idx, end_idx, alpha=0.3, color='green', label='Ground Truth')
        
        # Plot prediction
        if defect_prob[i] > threshold:
            start_idx = int(defect_start[i] * len(signal))
            end_idx = int(defect_end[i] * len(signal))
            ax.axvspan(start_idx, end_idx, alpha=0.3, color='red', label='Prediction')
            ax.set_title(f"Signal {i}: Defect Prob = {defect_prob[i]:.4f}")
        else:
            ax.set_title(f"Signal {i}: No Defect Detected (Prob = {defect_prob[i]:.4f})")
        
        # Add legend if needed
        if (labels_np[i] > 0.5) or (defect_prob[i] > threshold):
            ax.legend()
    
    plt.tight_layout()
    plt.show()


def test_on_json_file(model, json_file, seq_length=50):
    """
    Test the model on a JSON file
    
    Args:
        model: Trained model
        json_file: Path to JSON file
        seq_length: Number of signals per sequence
    """
    device = next(model.parameters()).device
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for scan_key, scan_data in data.items():
        print(f"Testing on scan: {scan_key}")
        
        signal_keys = sorted(scan_data.keys(),
                            key=lambda x: int(round(float(x.split('_')[0]))))
        
        signals = []
        labels = []
        defect_positions = []
        
        # Take only the first seq_length signals or pad if needed
        for i, signal_key in enumerate(signal_keys[:seq_length]):
            signal_info = scan_data[signal_key]
            signal = np.array(signal_info['signal'], dtype=np.float32)
            signals.append(signal)
            
            defect_name = signal_key.split('_')[1]
            if defect_name == 'Health':
                labels.append(0.0)
                defect_positions.append([0.0, 0.0])
            else:
                defect_range = signal_key.split('_')[2].split('-')
                defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                labels.append(1.0)
                defect_positions.append([defect_start, defect_end])
        
        if len(signals) < seq_length:
            pad_length = seq_length - len(signals)
            signal_length = len(signals[0])
            
            signals.extend([np.zeros(signal_length, dtype=np.float32) for _ in range(pad_length)])
            labels.extend([0.0 for _ in range(pad_length)])
            defect_positions.extend([[0.0, 0.0] for _ in range(pad_length)])
        
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        defect_positions_tensor = torch.tensor(defect_positions, dtype=torch.float32)
        
        visualize_predictions(model, signals_tensor, labels_tensor, defect_positions_tensor)
        
        user_input = input("Press Enter to continue to next scan, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break


def main():
    model_path = "models/improved_model_20250615_033905/best_model.pth"
    json_file = "json_data/WOT-D456_A4_003_Ch-0_D0.5-10.json"

    model, device = load_model(model_path)
    print(f"Model loaded on {device}")
    
    test_on_json_file(model, json_file)


if __name__ == "__main__":
    main()
