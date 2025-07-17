import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel
from defect_focused_dataset import get_defect_focused_dataloader


def load_model_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        val_accuracy = checkpoint.get('val_accuracy', 'Unknown')
        print(f"Loaded checkpoint from epoch {epoch} with validation accuracy: {val_accuracy}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    return model


def quick_test_model(model, test_loader, device, model_name):
    """Quick model testing with basic metrics"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    
    criterion = nn.BCELoss()
    
    print(f"Testing {model_name} model...")
    
    with torch.no_grad():
        for signals, labels, _ in tqdm(test_loader, desc="Testing"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            detection_prob = model(signals)
            loss = criterion(detection_prob, labels)
            
            # Convert to numpy
            preds = (detection_prob > 0.5).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_predictions.extend(preds.flatten())
            all_labels.extend(labels_np.flatten())
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    print(f"\n{model_name} Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Total samples: {len(all_labels)}")
    
    return {
        'model_name': model_name,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(all_labels)
    }


def find_best_model_checkpoint(models_dir, model_name):
    """Find the best model checkpoint for a given model"""
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith(f"{model_name}_")]
    
    if not model_dirs:
        print(f"No model directories found for {model_name}")
        return None
    
    # Get the most recent model directory
    model_dirs.sort(reverse=True)
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
    
    # Define models to test
    models_to_test = {
        "Simple": SimpleDetectionModel(signal_length=320),
        "Complex": ComplexDetectionModel(signal_length=320)
    }
    
    models_dir = "models"
    results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"Testing {model_name} Detection Model")
        print(f"{'='*50}")
        
        # Find best checkpoint
        checkpoint_path = find_best_model_checkpoint(models_dir, model_name)
        
        if checkpoint_path is None:
            print(f"No checkpoint found for {model_name} model. Skipping...")
            continue
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load and test model
        model = model.to(device)
        model = load_model_checkpoint(checkpoint_path, model, device)
        
        result = quick_test_model(model, test_loader, device, model_name)
        results[model_name] = result
    
    # Print summary
    print(f"\n{'='*50}")
    print("TESTING SUMMARY")
    print(f"{'='*50}")
    
    if results:
        print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"{model_name:<10} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<10.4f} {result['f1_score']:<10.4f}")
    else:
        print("No models were tested. Please check if model checkpoints exist.")


if __name__ == "__main__":
    main()
