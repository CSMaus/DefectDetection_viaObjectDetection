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

from hybrid_model import HybridModel
from defect_focused_dataset import get_defect_focused_dataloader
from realistic_noise_augmentation import RealisticNoiseAugmentation


def enhanced_position_loss(pred_start, pred_end, gt_start, gt_end, mask):
    """Enhanced position loss with IoU optimization"""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_start.device)
    
    pred_start_masked = pred_start[mask > 0.5]
    pred_end_masked = pred_end[mask > 0.5]
    gt_start_masked = gt_start[mask > 0.5]
    gt_end_masked = gt_end[mask > 0.5]
    
    # L1 loss
    l1_start = F.l1_loss(pred_start_masked, gt_start_masked)
    l1_end = F.l1_loss(pred_end_masked, gt_end_masked)
    l1_loss = (l1_start + l1_end) / 2
    
    # IoU loss
    pred_lengths = torch.abs(pred_end_masked - pred_start_masked)
    gt_lengths = torch.abs(gt_end_masked - gt_start_masked)
    
    overlap_starts = torch.maximum(pred_start_masked, gt_start_masked)
    overlap_ends = torch.minimum(pred_end_masked, gt_end_masked)
    overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
    
    unions = pred_lengths + gt_lengths - overlaps
    ious = overlaps / (unions + 1e-8)
    iou_loss = 1.0 - ious.mean()
    
    # Length preservation
    length_loss = F.l1_loss(pred_lengths, gt_lengths)
    
    # Consistency loss
    consistency_loss = F.relu(pred_start_masked - pred_end_masked + 0.01).mean()
    
    total_loss = (
        1.0 * l1_loss +
        2.0 * iou_loss +
        0.5 * length_loss +
        1.0 * consistency_loss
    )
    
    return total_loss


def train_hybrid_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Training strategy:
    Stage 1: Train position module only (keep proven detection frozen)
    Stage 2: Fine-tune both together
    """
    os.makedirs(save_dir, exist_ok=True)
    
    noise_augmenter = RealisticNoiseAugmentation(augment_probability=0.25)
    
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
        'learning_rate': [],
        'training_stage': []
    }
    
    cls_criterion = nn.BCELoss()
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training stages
        if epoch < 5:
            stage = "detection_only"
            model.unfreeze_detection_path()
            model.freeze_position_module()
            detection_weight = 1.0
            position_weight = 0.0  # Don't train position yet
        elif epoch < 15:
            stage = "position_only"
            model.freeze_detection_path()  # Keep proven detection frozen
            model.unfreeze_position_module()
            detection_weight = 0.0
            position_weight = 1.0
        else:
            stage = "joint_training"
            model.unfreeze_detection_path()
            model.unfreeze_position_module()
            detection_weight = 1.0
            position_weight = 1.0
        
        print(f"Epoch {epoch+1}/{num_epochs} - Stage: {stage}")
        
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_pos_loss = 0.0
        train_correct = 0
        train_total = 0
        train_pos_correct = 0
        train_pos_total = 0
        
        for signals, labels, defect_positions in tqdm(train_loader, desc="Training"):
            signals = signals.to(device)
            labels = labels.to(device)
            defect_positions = defect_positions.to(device)
            
            if model.training:
                signals = noise_augmenter.augment_sequence(signals)
            
            defect_prob, defect_start, defect_end = model(signals)
            
            # Classification loss
            if detection_weight > 0:
                cls_loss = cls_criterion(defect_prob, labels)
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            # Position loss
            gt_defect_mask = (labels > 0.5).float()
            
            if gt_defect_mask.sum() > 0 and position_weight > 0:
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
                        overlap_starts = torch.maximum(pred_starts_defect, gt_starts_defect)
                        overlap_ends = torch.minimum(pred_ends_defect, gt_ends_defect)
                        overlaps = torch.clamp(overlap_ends - overlap_starts, min=0)
                        
                        pred_lengths = torch.abs(pred_ends_defect - pred_starts_defect)
                        gt_lengths = torch.abs(gt_ends_defect - gt_starts_defect)
                        unions = pred_lengths + gt_lengths - overlaps
                        
                        ious = overlaps / (unions + 1e-8)
                        train_pos_correct += (ious >= 0.3).sum().item()
                        train_pos_total += len(ious)
            else:
                pos_loss = torch.tensor(0.0, device=device)
            
            loss = detection_weight * cls_loss + position_weight * pos_loss
            
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_pos_loss += pos_loss.item()
            
            with torch.no_grad():
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
                    pos_loss = enhanced_position_loss(
                        defect_start, defect_end,
                        defect_positions[:, :, 0], defect_positions[:, :, 1],
                        gt_defect_mask
                    )
                    
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
                
                loss = cls_loss + pos_loss
                
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
        history['training_stage'].append(stage)
        
        print(f"Train Acc: {train_accuracy:.4f}, Train Pos Acc: {train_pos_accuracy:.4f}")
        print(f"Val Acc: {val_accuracy:.4f}, Val Pos Acc: {val_pos_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'history': history
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"New best accuracy: {val_accuracy:.4f}!")
    
    return history


def main():
    signal_length = 320
    hidden_sizes = [128, 64, 32]
    num_heads = 8
    dropout = 0.15
    num_transformer_layers = 4
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.0008
    weight_decay = 0.01
    
    json_dir = "json_data_07/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"models/hybrid_model_{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Hybrid model training:")
    print("  - Keeps proven detection architecture (97% accuracy)")
    print("  - Adds separate position module")
    print("  - Stage 1: Train detection only (epochs 1-5)")
    print("  - Stage 2: Train position only (epochs 6-15)")
    print("  - Stage 3: Fine-tune both (epochs 16-20)")
    
    train_loader, val_loader = get_defect_focused_dataloader(
        json_dir, 
        batch_size=batch_size, 
        seq_length=30,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    model = HybridModel(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        dropout=dropout,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    
    history = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir
    )
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    print(f"Training complete! Best accuracy: {max(history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
