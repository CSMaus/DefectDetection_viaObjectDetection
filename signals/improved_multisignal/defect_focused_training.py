import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from improved_model import ImprovedMultiSignalClassifier
from defect_focused_dataset import get_defect_focused_dataloader


def train_defect_focused_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Train the model with focus on defect localization
    - Still outputs 3 values (defect_prob, defect_start, defect_end)
    - But trains position prediction ONLY on GT defective signals
    - Detection training uses all signals in the sequence
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data (defect-containing sequences only)
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save checkpoints
        
    Returns:
        dict: Training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'epochs': [],
        'train_loss': [],
        'train_cls_loss': [],
        'train_pos_loss': [],
        'train_accuracy': [],
        'train_pos_accuracy': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_pos_loss': [],
        'val_accuracy': [],
        'val_pos_accuracy': [],
        'learning_rate': []
    }
    
    cls_criterion = nn.BCELoss()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 7  # Increased patience for defect-focused training
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_pos_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pos_correct = 0
        train_pos_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for signals, labels, defect_positions in tqdm(train_loader, desc="Training"):
            signals = signals.to(device)
            labels = labels.to(device)
            defect_positions = defect_positions.to(device)
            
            defect_prob, defect_start, defect_end = model(signals)
            
            # Classification loss (on all signals)
            cls_loss = cls_criterion(defect_prob, labels)
            
            # Position loss ONLY on GT defective signals (key change!)
            gt_defect_mask = (labels > 0.5).float()  # Use GT labels, not predictions
            
            if gt_defect_mask.sum() > 0:
                # Use Smooth L1 Loss for position prediction, but ONLY on GT defective signals
                start_loss = F.smooth_l1_loss(
                    defect_start * gt_defect_mask, 
                    defect_positions[:, :, 0] * gt_defect_mask, 
                    reduction='sum'
                ) / (gt_defect_mask.sum() + 1e-8)
                
                end_loss = F.smooth_l1_loss(
                    defect_end * gt_defect_mask, 
                    defect_positions[:, :, 1] * gt_defect_mask, 
                    reduction='sum'
                ) / (gt_defect_mask.sum() + 1e-8)
                
                pos_loss = (start_loss + end_loss) / 2
                
                # Calculate position accuracy (IoU-based)
                with torch.no_grad():
                    pred_starts_defect = defect_start[gt_defect_mask > 0.5]
                    pred_ends_defect = defect_end[gt_defect_mask > 0.5]
                    gt_starts_defect = defect_positions[:, :, 0][gt_defect_mask > 0.5]
                    gt_ends_defect = defect_positions[:, :, 1][gt_defect_mask > 0.5]
                    
                    if len(pred_starts_defect) > 0:
                        # Calculate IoU for position accuracy
                        overlap_starts = torch.maximum(pred_starts_defect, gt_starts_defect)
                        overlap_ends = torch.minimum(pred_ends_defect, gt_ends_defect)
                        overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
                        
                        pred_lengths = torch.abs(pred_ends_defect - pred_starts_defect)
                        gt_lengths = torch.abs(gt_ends_defect - gt_starts_defect)
                        unions = pred_lengths + gt_lengths - overlaps
                        
                        ious = overlaps / (unions + 1e-8)
                        train_pos_correct += (ious >= 0.3).sum().item()  # IoU >= 0.3 threshold
                        train_pos_total += len(ious)
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            # Combined loss with higher weight on position for defect-focused training
            loss = cls_loss + 2.0 * pos_loss  # Increased position loss weight
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_pos_loss += pos_loss.item()
            train_correct += ((defect_prob > 0.5) == (labels > 0.5)).sum().item()
            train_total += labels.numel()
        
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_pos_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_pos_accuracy = train_pos_correct / max(train_pos_total, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_pos_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pos_correct = 0
        val_pos_total = 0
        
        with torch.no_grad():
            for signals, labels, defect_positions in tqdm(val_loader, desc="Validation"):
                signals = signals.to(device)
                labels = labels.to(device)
                defect_positions = defect_positions.to(device)
                
                defect_prob, defect_start, defect_end = model(signals)
                
                cls_loss = cls_criterion(defect_prob, labels)
                
                gt_defect_mask = (labels > 0.5).float()
                
                if gt_defect_mask.sum() > 0:
                    start_loss = F.smooth_l1_loss(
                        defect_start * gt_defect_mask, 
                        defect_positions[:, :, 0] * gt_defect_mask, 
                        reduction='sum'
                    ) / (gt_defect_mask.sum() + 1e-8)
                    
                    end_loss = F.smooth_l1_loss(
                        defect_end * gt_defect_mask, 
                        defect_positions[:, :, 1] * gt_defect_mask, 
                        reduction='sum'
                    ) / (gt_defect_mask.sum() + 1e-8)
                    
                    pos_loss = (start_loss + end_loss) / 2
                    
                    # Calculate validation position accuracy
                    pred_starts_defect = defect_start[gt_defect_mask > 0.5]
                    pred_ends_defect = defect_end[gt_defect_mask > 0.5]
                    gt_starts_defect = defect_positions[:, :, 0][gt_defect_mask > 0.5]
                    gt_ends_defect = defect_positions[:, :, 1][gt_defect_mask > 0.5]
                    
                    if len(pred_starts_defect) > 0:
                        overlap_starts = torch.maximum(pred_starts_defect, gt_starts_defect)
                        overlap_ends = torch.minimum(pred_ends_defect, gt_ends_defect)
                        overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
                        
                        pred_lengths = torch.abs(pred_ends_defect - pred_starts_defect)
                        gt_lengths = torch.abs(gt_ends_defect - gt_starts_defect)
                        unions = pred_lengths + gt_lengths - overlaps
                        
                        ious = overlaps / (unions + 1e-8)
                        val_pos_correct += (ious >= 0.3).sum().item()
                        val_pos_total += len(ious)
                else:
                    pos_loss = torch.tensor(0.0, device=device)
                
                loss = cls_loss + 2.0 * pos_loss
                
                val_loss += loss.item()
                val_cls_loss += cls_loss.item()
                val_pos_loss += pos_loss.item()
                val_correct += ((defect_prob > 0.5) == (labels > 0.5)).sum().item()
                val_total += labels.numel()
        
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_pos_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_pos_accuracy = val_pos_correct / max(val_pos_total, 1)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Update history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_pos_loss'].append(train_pos_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_pos_accuracy'].append(train_pos_accuracy)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_pos_loss'].append(val_pos_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_pos_accuracy'].append(val_pos_accuracy)
        history['learning_rate'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train Pos Acc: {train_pos_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Pos Acc: {val_pos_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }
        
        epoch_checkpoint_path = os.path.join(save_dir, f"{epoch+1}_cp_defect_focused_model.pth")
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1}")
        
        with open(os.path.join(save_dir, 'training_history_defect_focused.json'), 'w') as f:
            json.dump(history, f)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_defect_focused_model.pth'))
            patience_counter = 0
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    return history


def plot_defect_focused_history(history, save_path=None):
    """Plot training history with position accuracy"""
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['epochs'], history['train_cls_loss'], label='Train Cls Loss')
    plt.plot(history['epochs'], history['val_cls_loss'], label='Val Cls Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['epochs'], history['train_pos_loss'], label='Train Pos Loss')
    plt.plot(history['epochs'], history['val_pos_loss'], label='Val Pos Loss')
    plt.title('Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(history['epochs'], history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['epochs'], history['val_accuracy'], label='Val Accuracy')
    plt.title('Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(history['epochs'], history['train_pos_accuracy'], label='Train Pos Accuracy')
    plt.plot(history['epochs'], history['val_pos_accuracy'], label='Val Pos Accuracy')
    plt.title('Position Accuracy (IoU >= 0.3)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(history['epochs'], history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    signal_length = 320
    hidden_sizes = [128, 64, 32]
    num_heads = 8
    dropout = 0.2
    num_transformer_layers = 4
    batch_size = 8
    num_epochs = 15
    learning_rate = 0.0005
    weight_decay = 0.01
    
    json_dir = "json_data/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"models/defect_focused_model_{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get defect-focused train and validation loaders
    train_loader, val_loader = get_defect_focused_dataloader(
        json_dir, 
        batch_size=batch_size, 
        seq_length=30,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1  # At least 1 defect per sequence
    )
    
    model = ImprovedMultiSignalClassifier(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        dropout=dropout,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    history = train_defect_focused_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir
    )
    
    plot_defect_focused_history(history, save_path=os.path.join(save_dir, 'defect_focused_training_history.png'))
    
    with open(os.path.join(save_dir, 'defect_focused_training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # in separate script will export with correct version ONNX model
    # dummy_input = torch.randn(1, 50, signal_length).to(device)
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     os.path.join(save_dir, 'defect_focused_model.onnx'),
    #     export_params=True,
    #     opset_version=12,
    #     input_names=['input'],
    #     output_names=['defect_prob', 'defect_start', 'defect_end'],
    #     dynamic_axes={
    #         'input': {0: 'batch_size', 1: 'num_signals'},
    #         'defect_prob': {0: 'batch_size', 1: 'num_signals'},
    #         'defect_start': {0: 'batch_size', 1: 'num_signals'},
    #         'defect_end': {0: 'batch_size', 1: 'num_signals'},
    #     }
    # )
    #
    # print(f"Defect-focused model exported to {os.path.join(save_dir, 'defect_focused_model.onnx')}")


if __name__ == "__main__":
    main()
