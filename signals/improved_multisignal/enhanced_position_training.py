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

from enhanced_position_model_acc091 import EnhancedPositionMultiSignalClassifier
from defect_focused_dataset import get_defect_focused_dataloader
from realistic_noise_augmentation import RealisticNoiseAugmentation


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples in position prediction"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Calculate L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # Apply focal weighting (focus on hard examples)
        focal_weight = self.alpha * (l1_loss ** self.gamma)
        focal_loss = focal_weight * l1_loss
        
        return focal_loss.mean()


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


def train_enhanced_position_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Enhanced multi-stage training:
    Stage 1: Train detection head only (epochs 1-3)
    Stage 2: Train position head only (epochs 4-10) 
    Stage 3: Fine-tune both together (epochs 11-15)
    
    Includes realistic noise augmentation for robustness
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize realistic noise augmentation
    noise_augmenter = RealisticNoiseAugmentation(augment_probability=0.25)  # 25% of sequences get noise
    print(f"Realistic noise augmentation enabled: {noise_augmenter.augment_probability*100:.1f}% of training sequences will be augmented")
    
    history = {
        'epochs': [],
        'train_loss': [],
        'train_cls_loss': [],
        'train_pos_loss': [],
        'train_accuracy': [],
        'train_pos_accuracy': [],
        'train_pos_accuracy_strict': [],  # IoU >= 0.5
        'val_loss': [],
        'val_cls_loss': [],
        'val_pos_loss': [],
        'val_accuracy': [],
        'val_pos_accuracy': [],
        'val_pos_accuracy_strict': [],
        'learning_rate': [],
        'training_stage': []
    }
    
    cls_criterion = nn.BCELoss()
    
    # Early stopping parameters
    best_val_pos_accuracy = 0.0  # Track best position accuracy instead of loss
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # SIMPLIFIED TRAINING STAGES - Detection Priority
        if epoch < 3:
            stage = "detection_focus"
            # Don't freeze anything, just weight detection much higher
            detection_weight = 5.0  # Strong detection priority
            position_weight = 0.5   # Minimal position training
        else:
            stage = "balanced_training"
            # Balanced training but still prioritize detection
            detection_weight = 3.0  # Detection priority maintained
            position_weight = 1.0   # Position training enabled
        
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_pos_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pos_correct = 0
        train_pos_correct_strict = 0
        train_pos_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Stage: {stage} (Det:{detection_weight}x, Pos:{position_weight}x)")
        
        for signals, labels, defect_positions in tqdm(train_loader, desc="Training"):
            signals = signals.to(device)
            labels = labels.to(device)
            defect_positions = defect_positions.to(device)
            
            # Apply realistic noise augmentation during training (not validation)
            # This REPLACES original signals with noisy versions (not adding more data)
            if model.training:
                signals = noise_augmenter.augment_sequence(signals)
            
            defect_prob, defect_start, defect_end = model(signals)
            
            # Classification loss
            cls_loss = cls_criterion(defect_prob, labels)
            
            # Position loss ONLY on GT defective signals
            gt_defect_mask = (labels > 0.5).float()
            
            if gt_defect_mask.sum() > 0 and position_weight > 0:
                # Use enhanced position loss
                pos_loss = enhanced_position_loss(
                    defect_start, defect_end,
                    defect_positions[:, :, 0], defect_positions[:, :, 1],
                    gt_defect_mask
                )
                
                # Calculate position accuracy
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
                        train_pos_correct += (ious >= 0.3).sum().item()
                        train_pos_correct_strict += (ious >= 0.5).sum().item()
                        train_pos_total += len(ious)
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            # DETECTION-PRIORITY combined loss
            loss = detection_weight * cls_loss + position_weight * pos_loss
            
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
        train_pos_accuracy_strict = train_pos_correct_strict / max(train_pos_total, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_pos_loss = 0.0
        val_correct = 0
        val_total = 0
        val_pos_correct = 0
        val_pos_correct_strict = 0
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
                    pos_loss = enhanced_position_loss(
                        defect_start, defect_end,
                        defect_positions[:, :, 0], defect_positions[:, :, 1],
                        gt_defect_mask
                    )
                    
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
                        val_pos_correct_strict += (ious >= 0.5).sum().item()
                        val_pos_total += len(ious)
                else:
                    pos_loss = torch.tensor(0.0, device=device)
                
                loss = detection_weight * cls_loss + position_weight * pos_loss
                
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
        val_pos_accuracy_strict = val_pos_correct_strict / max(val_pos_total, 1)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Update history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_pos_loss'].append(train_pos_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_pos_accuracy'].append(train_pos_accuracy)
        history['train_pos_accuracy_strict'].append(train_pos_accuracy_strict)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_pos_loss'].append(val_pos_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_pos_accuracy'].append(val_pos_accuracy)
        history['val_pos_accuracy_strict'].append(val_pos_accuracy_strict)
        history['learning_rate'].append(current_lr)
        history['training_stage'].append(stage)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Train Pos Acc (IoU≥0.3): {train_pos_accuracy:.4f}, Train Pos Acc (IoU≥0.5): {train_pos_accuracy_strict:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"Val Pos Acc (IoU≥0.3): {val_pos_accuracy:.4f}, Val Pos Acc (IoU≥0.5): {val_pos_accuracy_strict:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_pos_accuracy': val_pos_accuracy,
            'val_pos_accuracy_strict': val_pos_accuracy_strict,
            'history': history
        }
        
        epoch_checkpoint_path = os.path.join(save_dir, f"{epoch+1}_cp_enhanced_position_model.pth")
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch+1}")
        
        with open(os.path.join(save_dir, 'enhanced_training_history.json'), 'w') as f:
            json.dump(history, f)
        
        # Track best position accuracy (more important than loss)
        if val_pos_accuracy > best_val_pos_accuracy:
            best_val_pos_accuracy = val_pos_accuracy
            torch.save(checkpoint, os.path.join(save_dir, 'best_enhanced_position_model.pth'))
            patience_counter = 0
            print(f"New best position accuracy: {val_pos_accuracy:.4f}!")
        else:
            patience_counter += 1
            print(f"No position improvement for {patience_counter} epochs")
        
        if patience_counter >= patience and epoch > 10:  # Only early stop after joint training starts
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    return history


def plot_enhanced_training_history(history, save_path=None):
    """Plot enhanced training history with stages"""
    plt.figure(figsize=(20, 15))
    
    # Color code by training stage
    stage_colors = {'detection_only': 'red', 'position_only': 'blue', 'joint_training': 'green'}
    colors = [stage_colors.get(stage, 'black') for stage in history['training_stage']]
    
    plt.subplot(3, 3, 1)
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['epochs'], history['val_loss'], label='Val Loss', color='orange')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 2)
    plt.plot(history['epochs'], history['train_cls_loss'], label='Train Cls Loss', color='blue')
    plt.plot(history['epochs'], history['val_cls_loss'], label='Val Cls Loss', color='orange')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
    plt.plot(history['epochs'], history['train_pos_loss'], label='Train Pos Loss', color='blue')
    plt.plot(history['epochs'], history['val_pos_loss'], label='Val Pos Loss', color='orange')
    plt.title('Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 4)
    plt.plot(history['epochs'], history['train_accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['epochs'], history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 5)
    plt.plot(history['epochs'], history['train_pos_accuracy'], label='Train Pos Acc (IoU≥0.3)', color='blue')
    plt.plot(history['epochs'], history['val_pos_accuracy'], label='Val Pos Acc (IoU≥0.3)', color='orange')
    plt.title('Position Accuracy (IoU ≥ 0.3)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 6)
    plt.plot(history['epochs'], history['train_pos_accuracy_strict'], label='Train Pos Acc (IoU≥0.5)', color='blue')
    plt.plot(history['epochs'], history['val_pos_accuracy_strict'], label='Val Pos Acc (IoU≥0.5)', color='orange')
    plt.title('Position Accuracy (IoU ≥ 0.5) - STRICT')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 3, 7)
    plt.plot(history['epochs'], history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(3, 3, 8)
    # Training stage visualization
    stage_mapping = {
        'detection_focus': 1, 
        'balanced_training': 2,
        # Legacy stage names (in case they exist)
        'detection_only': 1, 
        'position_only': 2, 
        'joint_training': 3
    }
    stage_nums = [stage_mapping.get(stage, 1) for stage in history['training_stage']]
    plt.plot(history['epochs'], stage_nums, marker='o')
    plt.title('Training Stage')
    plt.xlabel('Epoch')
    plt.ylabel('Stage')
    plt.yticks([1, 2], ['Detection Focus', 'Balanced Training'])
    plt.grid(True)
    
    plt.subplot(3, 3, 9)
    # Position accuracy comparison
    plt.plot(history['epochs'], history['val_pos_accuracy'], label='IoU ≥ 0.3', color='blue')
    plt.plot(history['epochs'], history['val_pos_accuracy_strict'], label='IoU ≥ 0.5', color='red')
    plt.title('Validation Position Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main():
    signal_length = 320
    hidden_sizes = [128, 64, 32]
    num_heads = 8
    dropout = 0.15
    num_transformer_layers = 4
    batch_size = 8
    num_epochs = 25
    learning_rate = 0.0008
    weight_decay = 0.01
    
    json_dir = "json_data_0718/"  #"json_data_07/"  # json_data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"models/enhanced_position_model_{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Enhanced training with:")
    print("  - Multi-stage training (detection → position → joint)")
    print("  - Realistic noise augmentation (25% of sequences)")
    print("  - Enhanced position loss (L1 + IoU + consistency)")
    print("  - Dedicated position regression head")
    
    # Get defect-focused train and validation loaders
    train_loader, val_loader = get_defect_focused_dataloader(
        json_dir, 
        batch_size=batch_size, 
        seq_length=30,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    model = EnhancedPositionMultiSignalClassifier(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        dropout=dropout,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enhanced model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    
    history = train_enhanced_position_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir
    )
    
    plot_enhanced_training_history(history, save_path=os.path.join(save_dir, 'enhanced_training_history.png'))
    
    with open(os.path.join(save_dir, 'enhanced_training_history.json'), 'w') as f:
        json.dump(history, f)
    
    print(f"Enhanced position model training complete!")
    print(f"Best validation position accuracy: {max(history['val_pos_accuracy']):.4f}")
    print(f"Best validation strict position accuracy: {max(history['val_pos_accuracy_strict']):.4f}")


if __name__ == "__main__":
    main()
