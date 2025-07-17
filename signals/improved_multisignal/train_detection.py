import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from detection_models.simple_detection_model import SimpleDetectionModel
from detection_models.complex_detection_model import ComplexDetectionModel
from detection_models.complex_onnx import ComplexDetectionModelONNX
from detection_models.complex_fix import ComplexDetectionModelFix
from detection_models.noise_robust_tr2 import NoiseRobustDetectionModel
from detection_models.pattern_embedding import PatternEmbeddingModel
from detection_models.enhanced_pattern import EnhancedPatternModel

from defect_focused_dataset import get_defect_focused_dataloader


def plot_training_history(history, save_path=None):
    """Plot training history similar to enhanced_position_training.py"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['epochs'], history['train_accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['epochs'], history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['epochs'], history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(2, 3, 4)
    # Best accuracy line
    best_val_acc = max(history['val_accuracy'])
    plt.axhline(y=best_val_acc, color='red', linestyle='--', label=f'Best Val Acc: {best_val_acc:.4f}')
    plt.plot(history['epochs'], history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Validation Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
    plt.plot(history['epochs'], loss_diff, label='|Train Loss - Val Loss|', color='purple')
    plt.title('Train-Val Loss Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # Accuracy difference
    acc_diff = [abs(t - v) for t, v in zip(history['train_accuracy'], history['val_accuracy'])]
    plt.plot(history['epochs'], acc_diff, label='|Train Acc - Val Acc|', color='green')
    plt.title('Train-Val Accuracy Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def train_detection_model(model, train_loader, val_loader, num_epochs, device, model_name, save_dir):
    """Enhanced detection training with checkpoints and history tracking"""
    
    # Create model-specific directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_save_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.BCELoss()
    
    best_accuracy = 0.0
    
    # Initialize training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for signals, labels, _ in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Forward pass
            detection_prob = model(signals)
            loss = criterion(detection_prob, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_correct += ((detection_prob > 0.5) == (labels > 0.5)).sum().item()
            train_total += labels.numel()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels, _ in tqdm(val_loader, desc="Validation"):
                signals = signals.to(device)
                labels = labels.to(device)
                
                detection_prob = model(signals)
                loss = criterion(detection_prob, labels)
                
                val_loss += loss.item()
                val_correct += ((detection_prob > 0.5) == (labels > 0.5)).sum().item()
                val_total += labels.numel()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Update history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'history': history
        }
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(model_save_dir, f"epoch_{epoch+1:02d}_checkpoint.pth")
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Save training history JSON
        history_path = os.path.join(model_save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(model_save_dir, f"best_detection.pth")
            torch.save(checkpoint, best_model_path)
            print(f"  New best accuracy: {val_accuracy:.4f}! Saved to {best_model_path}")
        
        print()
    
    plot_path = os.path.join(model_save_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print(f"Training completed for {model_name}")
    print(f"All checkpoints and history saved in: {model_save_dir}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    return best_accuracy, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create main models directory
    main_models_dir = "models"
    os.makedirs(main_models_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader = get_defect_focused_dataloader(
        "json_data_0717", # "json_data_0716/",
        batch_size=8,
        seq_length=50,  # 30,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    models = {
        # "ComplexONNX": ComplexDetectionModelONNX(signal_length=320)
        # "ComplexFix": ComplexDetectionModelFix(signal_length=320)
        # "NoiseRobust": NoiseRobustDetectionModel(signal_length=320)
        # "PatternEmbedding": PatternEmbeddingModel(signal_length=320)
        "EnhancedPattern": EnhancedPatternModel(signal_length=320)
    }
    
    results = {}
    all_histories = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name} Detection Model")
        print(f"{'='*50}")
        
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        best_acc, history = train_detection_model(
            model, train_loader, val_loader, 
            num_epochs=20, device=device, model_name=name, save_dir=main_models_dir
        )
        
        results[name] = best_acc
        all_histories[name] = history
        print(f"{name} Model Best Accuracy: {best_acc:.4f}")
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for name, acc in results.items():
        print(f"{name} Detection Model: {acc:.4f}")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    combined_results = {
        'timestamp': timestamp,
        'results': results,
        'histories': all_histories,
        'device': str(device),
        'total_epochs': 15
    }
    
    results_path = os.path.join(main_models_dir, f"detection_training_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to: {results_path}")


if __name__ == "__main__":
    main()
