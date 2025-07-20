import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from detection_models.position_localization import PositionLocalizationModel
from defect_focused_dataset import get_defect_focused_dataloader


def enhanced_position_loss(pred_start, pred_end, gt_start, gt_end, mask):
    """
    Enhanced position loss with multiple components:
    1. L1 loss for basic position accuracy
    2. IoU loss for overlap optimization
    3. Consistency loss to ensure start < end
    4. Length preservation loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_start.device)
    
    # Apply mask
    pred_start_masked = pred_start[mask > 0.5]
    pred_end_masked = pred_end[mask > 0.5]
    gt_start_masked = gt_start[mask > 0.5]
    gt_end_masked = gt_end[mask > 0.5]
    
    # 1. Basic L1 loss
    l1_start = F.l1_loss(pred_start_masked, gt_start_masked)
    l1_end = F.l1_loss(pred_end_masked, gt_end_masked)
    l1_loss = (l1_start + l1_end) / 2
    
    # 2. IoU loss (maximize overlap)
    pred_lengths = torch.abs(pred_end_masked - pred_start_masked)
    gt_lengths = torch.abs(gt_end_masked - gt_start_masked)
    
    overlap_starts = torch.maximum(pred_start_masked, gt_start_masked)
    overlap_ends = torch.minimum(pred_end_masked, gt_end_masked)
    overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
    
    unions = pred_lengths + gt_lengths - overlaps
    ious = overlaps / (unions + 1e-8)
    iou_loss = 1.0 - ious.mean()  # Maximize IoU
    
    # 3. Length preservation loss
    length_loss = F.l1_loss(pred_lengths, gt_lengths)
    
    # 4. Consistency loss (ensure start < end)
    consistency_loss = F.relu(pred_start_masked - pred_end_masked + 0.01).mean()
    
    # Combine losses
    total_loss = (
        1.0 * l1_loss +           # Basic position accuracy
        2.0 * iou_loss +          # Overlap optimization (most important)
        0.5 * length_loss +       # Length preservation
        1.0 * consistency_loss    # Consistency constraint
    )
    
    return total_loss


def calculate_position_accuracy(pred_start, pred_end, true_start, true_end):
    """Calculate accuracy based on IoU threshold (more realistic than simple tolerance)"""
    # Calculate IoU for each prediction
    pred_lengths = torch.abs(pred_end - pred_start)
    gt_lengths = torch.abs(true_end - true_start)
    
    overlap_starts = torch.maximum(pred_start, true_start)
    overlap_ends = torch.minimum(pred_end, true_end)
    overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
    
    unions = pred_lengths + gt_lengths - overlaps
    ious = overlaps / (unions + 1e-8)
    
    # Consider prediction accurate if IoU > 0.5
    accurate = ious > 0.5
    return accurate.float().mean().item()


def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(15, 10))
    
    # Loss plots
    plt.subplot(2, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plots
    plt.subplot(2, 3, 2)
    plt.plot(history['epochs'], history['train_accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history['epochs'], history['val_accuracy'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
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
    
    # Create model-specific directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_save_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.MSELoss()  # Mean Squared Error for position regression
    
    best_loss = float('inf')
    
    # Initialize training history
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
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_batches = 0
        
        for signals, labels, positions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            signals = signals.to(device)
            labels = labels.to(device)
            positions = positions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_start, pred_end = model(signals)
            
            # Calculate loss and accuracy ONLY for defective signals
            batch_loss = 0.0
            batch_accuracy = 0.0
            valid_samples = 0
            
            for batch_idx in range(signals.size(0)):
                # Find defective signals in this sequence
                defective_mask = labels[batch_idx] > 0  # Signals with defects
                
                if defective_mask.sum() > 0:  # If there are defective signals
                    # Calculate enhanced loss and accuracy
                    mask = labels[batch_idx]  # Defect mask
                    loss = enhanced_position_loss(
                        pred_start[batch_idx], pred_end[batch_idx],
                        positions[batch_idx, :, 0], positions[batch_idx, :, 1],
                        mask
                    )
                    
                    if mask.sum() > 0:  # Only calculate accuracy if there are defects
                        accuracy = calculate_position_accuracy(
                            pred_start[batch_idx, mask > 0.5], pred_end[batch_idx, mask > 0.5],
                            positions[batch_idx, mask > 0.5, 0], positions[batch_idx, mask > 0.5, 1]
                        )
                        batch_accuracy += accuracy
                    
                    batch_loss += loss
                    valid_samples += 1
            
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_accuracy = batch_accuracy / valid_samples
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += batch_loss.item()
                train_accuracy += batch_accuracy
                train_batches += 1
        
        train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_accuracy = train_accuracy / train_batches if train_batches > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for signals, labels, positions in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                positions = positions.to(device)
                
                # Forward pass
                pred_start, pred_end = model(signals)
                
                # Calculate loss and accuracy ONLY for defective signals
                batch_loss = 0.0
                batch_accuracy = 0.0
                valid_samples = 0
                
                for batch_idx in range(signals.size(0)):
                    # Find defective signals in this sequence
                    defective_mask = labels[batch_idx] > 0
                    
                    if defective_mask.sum() > 0:
                        # Calculate enhanced loss and accuracy
                        mask = labels[batch_idx]  # Defect mask
                        loss = enhanced_position_loss(
                            pred_start[batch_idx], pred_end[batch_idx],
                            positions[batch_idx, :, 0], positions[batch_idx, :, 1],
                            mask
                        )
                    
                    if mask.sum() > 0:  # Only calculate accuracy if there are defects
                        accuracy = calculate_position_accuracy(
                            pred_start[batch_idx, mask > 0.5], pred_end[batch_idx, mask > 0.5],
                            positions[batch_idx, mask > 0.5, 0], positions[batch_idx, mask > 0.5, 1]
                        )
                        batch_accuracy += accuracy
                    
                    batch_loss += loss
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
        
        # Update history
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
        
        # Save checkpoint every epoch
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
        
        # Save training history JSON
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
    
    # Create main models directory
    main_models_dir = "models"
    os.makedirs(main_models_dir, exist_ok=True)
    
    # Load data with ONLY defective sequences (isOnlyDefective=True)
    print("Loading ONLY defective sequences for position training...")
    train_loader, val_loader = get_defect_focused_dataloader(
        "json_data_0717",
        batch_size=8,
        seq_length=50,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1,
        isOnlyDefective=True  # Only defective sequences
    )
    
    # Create enhanced position localization model
    model = PositionLocalizationModel(
        signal_length=320,
        hidden_sizes=[128, 64, 32],  # Same as enhanced model
        num_heads=8,
        dropout=0.1,
        num_transformer_layers=4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train the model
    best_loss, history = train_position_model(
        model, train_loader, val_loader,
        num_epochs=20, device=device, 
        model_name="PositionLocalization", 
        save_dir=main_models_dir
    )
    
    print(f"Position Localization Model Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
