import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm
from datetime import datetime
import seaborn as sns

from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel
from defect_focused_dataset import get_defect_focused_dataloader


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        val_accuracy = checkpoint.get('val_accuracy', 'Unknown')
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy: {val_accuracy}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    return model


def evaluate_model(model, test_loader, device, model_name):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.BCELoss()
    
    print(f"Evaluating {model_name} model...")
    
    with torch.no_grad():
        for signals, labels, _ in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            detection_prob = model(signals)
            loss = criterion(detection_prob, labels)
            
            # Convert to numpy for analysis
            probs = detection_prob.cpu().numpy()
            preds = (detection_prob > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probabilities.extend(probs.flatten())
            all_predictions.extend(preds.flatten())
            all_labels.extend(labels_np.flatten())
            
            test_loss += loss.item()
            correct += ((detection_prob > 0.5) == (labels > 0.5)).sum().item()
            total += labels.numel()
    
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'predictions': all_predictions.tolist(),
        'probabilities': all_probabilities.tolist(),
        'labels': all_labels.tolist()
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    
    return results


def plot_evaluation_results(results, save_dir):
    """Create comprehensive evaluation plots"""
    model_name = results['model_name']
    predictions = np.array(results['predictions'])
    probabilities = np.array(results['probabilities'])
    labels = np.array(results['labels'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} Detection Model - Evaluation Results', fontsize=16)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # 3. Probability Distribution
    axes[0, 2].hist(probabilities[labels == 0], bins=30, alpha=0.7, label='No Defect', color='blue')
    axes[0, 2].hist(probabilities[labels == 1], bins=30, alpha=0.7, label='Defect', color='red')
    axes[0, 2].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[0, 2].set_xlabel('Prediction Probability')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Probability Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. Prediction vs Ground Truth
    sample_indices = np.arange(min(1000, len(labels)))  # Show first 1000 samples
    axes[1, 0].scatter(sample_indices, labels[:len(sample_indices)], alpha=0.6, label='Ground Truth', s=10)
    axes[1, 0].scatter(sample_indices, predictions[:len(sample_indices)], alpha=0.6, label='Predictions', s=10)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Label')
    axes[1, 0].set_title('Predictions vs Ground Truth (First 1000 samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5. Probability vs Ground Truth
    axes[1, 1].scatter(sample_indices, probabilities[:len(sample_indices)], 
                      c=labels[:len(sample_indices)], cmap='coolwarm', alpha=0.6, s=10)
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Prediction Probability')
    axes[1, 1].set_title('Prediction Probabilities (First 1000 samples)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 6. Performance Metrics Summary
    axes[1, 2].axis('off')
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_text = f"""
    Performance Metrics:
    
    Accuracy: {results['accuracy']:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1-Score: {f1:.4f}
    Specificity: {specificity:.4f}
    AUC: {roc_auc:.4f}
    
    Confusion Matrix:
    True Negatives: {tn}
    False Positives: {fp}
    False Negatives: {fn}
    True Positives: {tp}
    
    Total Samples: {results['total_samples']}
    Test Loss: {results['test_loss']:.4f}
    """
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{model_name.lower()}_evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved to: {plot_path}")
    
    # Save detailed metrics
    detailed_metrics = {
        'accuracy': results['accuracy'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'auc': roc_auc,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'test_loss': results['test_loss'],
        'total_samples': results['total_samples']
    }
    
    return detailed_metrics


def find_best_model_checkpoint(models_dir, model_name):
    """Find the best model checkpoint for a given model"""
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith(f"{model_name}_")]
    
    if not model_dirs:
        print(f"No model directories found for {model_name}")
        return None
    
    # Get the most recent model directory
    model_dirs.sort(reverse=True)  # Most recent first
    latest_model_dir = os.path.join(models_dir, model_dirs[0])
    
    # Look for best model checkpoint
    best_checkpoint = os.path.join(latest_model_dir, f"best_{model_name.lower()}_detection.pth")
    
    if os.path.exists(best_checkpoint):
        return best_checkpoint
    
    # If no best checkpoint, look for any checkpoint
    checkpoints = [f for f in os.listdir(latest_model_dir) if f.endswith('.pth')]
    if checkpoints:
        return os.path.join(latest_model_dir, checkpoints[0])
    
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = f"evaluation_results_detection_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    train_loader, val_loader = get_defect_focused_dataloader(
        "json_data_0717",
        batch_size=16, 
        seq_length=50,
        shuffle=False,  # Don't shuffle for consistent testing
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    # Use validation loader as test loader
    test_loader = val_loader
    
    # Define models to test
    models_to_test = {
        "Simple": SimpleDetectionModel(signal_length=320),
        "Complex": ComplexDetectionModel(signal_length=320)
    }
    
    models_dir = "models"
    all_results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_name} Detection Model")
        print(f"{'='*60}")
        
        # Find best checkpoint
        checkpoint_path = find_best_model_checkpoint(models_dir, model_name)
        
        if checkpoint_path is None:
            print(f"No checkpoint found for {model_name} model. Skipping...")
            continue
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load model
        model = model.to(device)
        model = load_model_checkpoint(checkpoint_path, model, device)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device, model_name)
        
        # Create evaluation plots and get detailed metrics
        detailed_metrics = plot_evaluation_results(results, results_dir)
        
        # Store results
        all_results[model_name] = {
            'checkpoint_path': checkpoint_path,
            'results': results,
            'detailed_metrics': detailed_metrics
        }
        
        print(f"\nDetailed metrics for {model_name}:")
        for metric, value in detailed_metrics.items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Save comprehensive results
    comprehensive_results = {
        'timestamp': timestamp,
        'device': str(device),
        'test_data_info': {
            'data_dir': "json_data_0717",
            'batch_size': 16,
            'seq_length': 50,
            'validation_split': 0.2
        },
        'results': all_results
    }
    
    results_file = os.path.join(results_dir, 'comprehensive_test_results.json')
    with open(results_file, 'w') as f:
        # Remove non-serializable data for JSON
        json_results = comprehensive_results.copy()
        for model_name in json_results['results']:
            # Remove large arrays from JSON (keep only metrics)
            json_results['results'][model_name]['results'] = {
                'model_name': json_results['results'][model_name]['results']['model_name'],
                'test_loss': json_results['results'][model_name]['results']['test_loss'],
                'accuracy': json_results['results'][model_name]['results']['accuracy'],
                'total_samples': json_results['results'][model_name]['results']['total_samples'],
                'correct_predictions': json_results['results'][model_name]['results']['correct_predictions']
            }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in: {results_dir}")
    print(f"Comprehensive results: {results_file}")
    
    # Print summary
    print("\nSUMMARY:")
    for model_name, data in all_results.items():
        metrics = data['detailed_metrics']
        print(f"{model_name} Model:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
