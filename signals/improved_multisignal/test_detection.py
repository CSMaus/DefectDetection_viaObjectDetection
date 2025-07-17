import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel
from defect_focused_dataset import get_defect_focused_dataloader


def test_detection(checkpoint_path, model_type="Complex"):
    """Test a single detection model with specified checkpoint path"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    _, test_loader = get_defect_focused_dataloader(
        "json_data_0717",
        batch_size=16, 
        seq_length=50,
        shuffle=False,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    # Initialize model
    if model_type == "Simple":
        model = SimpleDetectionModel(signal_length=320)
    elif model_type == "Complex":
        model = ComplexDetectionModel(signal_length=320)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        val_accuracy = checkpoint.get('val_accuracy', 'Unknown')
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy: {val_accuracy}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    # Test model
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    test_loss = 0.0
    
    criterion = nn.BCELoss()
    
    print(f"Testing {model_type} model...")
    
    with torch.no_grad():
        for signals, labels, _ in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            detection_prob = model(signals)
            loss = criterion(detection_prob, labels)
            
            # Store results
            all_probabilities.extend(detection_prob.cpu().numpy().flatten())
            all_predictions.extend((detection_prob > 0.5).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{model_type} Detection Model Test Results")
    print(f"{'='*50}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Total samples: {len(all_labels)}")
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # Create results visualization and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = f"test_results_{model_type}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Confusion matrix
    plt.subplot(1, 3, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    
    # Probability distribution
    plt.subplot(1, 3, 2)
    plt.hist(all_probabilities[all_labels == 0], bins=30, alpha=0.7, label='No Defect', color='blue')
    plt.hist(all_probabilities[all_labels == 1], bins=30, alpha=0.7, label='Defect', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sample predictions
    plt.subplot(1, 3, 3)
    sample_size = min(200, len(all_labels))
    indices = np.arange(sample_size)
    plt.scatter(indices, all_labels[:sample_size], alpha=0.6, label='Ground Truth', s=15, color='blue')
    plt.scatter(indices, all_predictions[:sample_size], alpha=0.6, label='Predictions', s=15, color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title(f'Predictions vs Ground Truth (First {sample_size} samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(results_dir, f"{model_type}_test_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTest results plot saved to: {plot_path}")
    
    # Save metrics as text file
    metrics_path = os.path.join(results_dir, f"{model_type}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"{model_type} Detection Model Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n")
    
    print(f"Test metrics saved to: {metrics_path}")
    print(f"All results saved in directory: {results_dir}")
    
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'test_loss': test_loss,
        'confusion_matrix': cm,
        'results_dir': results_dir
    }


if __name__ == "__main__":
    # SPECIFY YOUR CHECKPOINT PATH HERE
    checkpoint_path = "models/Complex_20250717_0800/best_complex_detection.pth"  # CHANGE THIS PATH
    model_type = "Complex"  # "Simple" or "Complex"
    
    print(f"Testing {model_type} model with checkpoint: {checkpoint_path}")
    
    try:
        results = test_detection(checkpoint_path, model_type)
        print(f"\nTesting completed successfully!")
        print(f"Results saved in: {results['results_dir']}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with the correct path.")
    except Exception as e:
        print(f"ERROR: {e}")
