import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from detection_models.position_localization import PositionLocalizationModel
from defect_focused_dataset import get_defect_focused_dataloader


def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['epochs'], history['train_accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history['epochs'], history['val_accuracy'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['epochs'], history['learning_rate'], 'g-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()


def train_position_model(model, train_loader, val_loader, num_epochs, device, model_name, save_dir):
    """Train position localization model using only defective signals"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_save_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    def calculate_position_accuracy(pred_start, pred_end, true_start, true_end, tolerance=0.1):
        """Calculate accuracy based on position tolerance"""
        start_correct = torch.abs(pred_start - true_start) <= tolerance
        end_correct = torch.abs(pred_end - true_end) <= tolerance
        both_correct = start_correct & end_correct
        return both_correct.float().mean().item()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_batches = 0
        
        for signals, labels, positions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            signals = signals.to(device)
            labels = labels.to(device)
            positions = positions.to(device)
            
            optimizer.zero_grad()
            
            pred_start, pred_end = model(signals)
            
            batch_loss = 0.0
            batch_accuracy = 0.0
            valid_samples = 0
            
            for batch_idx in range(signals.size(0)):
                defective_mask = labels[batch_idx] > 0
                
                if defective_mask.sum() > 0:
                    true_start = positions[batch_idx, defective_mask, 0]
                    true_end = positions[batch_idx, defective_mask, 1]
                    
                    pred_start_defective = pred_start[batch_idx, defective_mask]
                    pred_end_defective = pred_end[batch_idx, defective_mask]
                    
                    start_loss = criterion(pred_start_defective, true_start)
                    end_loss = criterion(pred_end_defective, true_end)
                    
                    accuracy = calculate_position_accuracy(pred_start_defective, pred_end_defective,
                                                         true_start, true_end)
                    
                    batch_loss += (start_loss + end_loss)
                    batch_accuracy += accuracy
                    valid_samples += 1
            
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_accuracy = batch_accuracy / valid_samples
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += batch_loss.item()
                train_accuracy += batch_accuracy
                train_batches += 1
        
        train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_accuracy = train_accuracy / train_batches if train_batches > 0 else 0.0
        
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for signals, labels, positions in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                positions = positions.to(device)
                
                pred_start, pred_end = model(signals)
                
                batch_loss = 0.0
                batch_accuracy = 0.0
                valid_samples = 0
                
                for batch_idx in range(signals.size(0)):
                    defective_mask = labels[batch_idx] > 0
                    
                    if defective_mask.sum() > 0:
                        true_start = positions[batch_idx, defective_mask, 0]
                        true_end = positions[batch_idx, defective_mask, 1]
                        
                        pred_start_defective = pred_start[batch_idx, defective_mask]
                        pred_end_defective = pred_end[batch_idx, defective_mask]
                        
                        start_loss = criterion(pred_start_defective, true_start)
                        end_loss = criterion(pred_end_defective, true_end)
                        
                        accuracy = calculate_position_accuracy(pred_start_defective, pred_end_defective,
                                                             true_start, true_end)
                        
                        batch_loss += (start_loss + end_loss)
                        batch_accuracy += accuracy
                        valid_samples += 1
                
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    batch_accuracy = batch_accuracy / valid_samples
                    val_loss += batch_loss.item()
                    val_accuracy += batch_accuracy
                    val_batches += 1
        
        val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_accuracy = val_accuracy / val_batches if val_batches > 0 else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_accuracy:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        }
        
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        history_path = os.path.join(model_save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(model_save_dir, f"best_position_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"  New best loss: {val_loss:.4f}! Saved to {best_model_path}")
        
        print()
    
    plot_path = os.path.join(model_save_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    print(f"Training completed for {model_name}")
    print(f"All checkpoints and history saved in: {model_save_dir}")
    print(f"Best validation loss: {best_loss:.4f}")
    
    return best_loss, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main_models_dir = "models"
    os.makedirs(main_models_dir, exist_ok=True)
    
    print("Loading ONLY defective sequences for position training...")
    train_loader, val_loader = get_defect_focused_dataloader(
        "json_data_0717",
        batch_size=8,
        seq_length=50,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1,
        isOnlyDefective=True
    )
    
    model = PositionLocalizationModel(
        signal_length=320,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    best_loss, history = train_position_model(
        model, train_loader, val_loader,
        num_epochs=15, device=device,
        model_name="PositionLocalization", 
        save_dir=main_models_dir
    )
    
    print(f"Position Localization Model Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
