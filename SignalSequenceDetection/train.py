import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime

from dataset_preparation import SignalSequenceDataset
from model import SignalSequenceDetector


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=10,
    device='cuda',
    save_dir='checkpoints',
    log_interval=10
):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_interval: Interval for logging
        
    Returns:
        dict: Training history
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_loss': [],
        'train_bbox_loss': [],
        'train_anomaly_loss': [],
        'val_cls_loss': [],
        'val_bbox_loss': [],
        'val_anomaly_loss': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_bbox_loss = 0.0
        train_anomaly_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Get data
            signals = batch['signals'].to(device)
            targets = batch['targets']
            
            # Forward pass
            optimizer.zero_grad()
            loss, loss_dict = model(signals, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_cls_loss += loss_dict['cls_loss'].item()
            train_bbox_loss += loss_dict['bbox_loss'].item()
            train_anomaly_loss += loss_dict['anomaly_consistency_loss'].item()
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Cls: {loss_dict['cls_loss'].item():.4f}, "
                      f"Bbox: {loss_dict['bbox_loss'].item():.4f}, "
                      f"Anomaly: {loss_dict['anomaly_consistency_loss'].item():.4f}")
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_bbox_loss /= len(train_loader)
        train_anomaly_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_bbox_loss = 0.0
        val_anomaly_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                signals = batch['signals'].to(device)
                targets = batch['targets']
                
                # Forward pass
                loss, loss_dict = model(signals, targets)
                
                # Update metrics
                val_loss += loss.item()
                val_cls_loss += loss_dict['cls_loss'].item()
                val_bbox_loss += loss_dict['bbox_loss'].item()
                val_anomaly_loss += loss_dict['anomaly_consistency_loss'].item()
        
        # Calculate average losses
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_bbox_loss /= len(val_loader)
        val_anomaly_loss /= len(val_loader)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_bbox_loss'].append(train_bbox_loss)
        history['train_anomaly_loss'].append(train_anomaly_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_bbox_loss'].append(val_bbox_loss)
        history['val_anomaly_loss'].append(val_anomaly_loss)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        print(f"  Train - Cls: {train_cls_loss:.4f}, Bbox: {train_bbox_loss:.4f}, Anomaly: {train_anomaly_loss:.4f}")
        print(f"  Val - Cls: {val_cls_loss:.4f}, Bbox: {val_bbox_loss:.4f}, Anomaly: {val_anomaly_loss:.4f}")
        
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
        
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print("  Saved best model!")
        
        # Save latest model
        torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pth'))
        
        # Save history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    return history


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history
        save_path: Path to save the plot
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot(history['train_loss'], label='Train')
    axs[0, 0].plot(history['val_loss'], label='Validation')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot classification loss
    axs[0, 1].plot(history['train_cls_loss'], label='Train')
    axs[0, 1].plot(history['val_cls_loss'], label='Validation')
    axs[0, 1].set_title('Classification Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot bounding box loss
    axs[1, 0].plot(history['train_bbox_loss'], label='Train')
    axs[1, 0].plot(history['val_bbox_loss'], label='Validation')
    axs[1, 0].set_title('Bounding Box Loss')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot anomaly consistency loss
    axs[1, 1].plot(history['train_anomaly_loss'], label='Train')
    axs[1, 1].plot(history['val_anomaly_loss'], label='Validation')
    axs[1, 1].set_title('Anomaly Consistency Loss')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get data
            signals = batch['signals'].to(device)
            targets = batch['targets']
            
            # Get predictions
            preds = model.predict(signals)
            
            # Store predictions and targets
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    return metrics


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predictions
        targets: List of targets
        
    Returns:
        dict: Metrics
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Calculate IoU for each prediction
    ious = []
    
    for preds, tgts in zip(predictions, targets):
        # Extract target defects
        target_defects = []
        for t in tgts:
            if t['label'] > 0:  # Not Health
                target_defects.append({
                    'position': t['position'] if 'position' in t else 0,
                    'class': t['label'],
                    'defect_position': t['bbox'][2:4].cpu().numpy() if torch.is_tensor(t['bbox']) else t['bbox'][2:4]
                })
        
        # Match predictions to targets
        matched_targets = set()
        
        for pred in preds:
            matched = False
            
            for i, target in enumerate(target_defects):
                if i in matched_targets:
                    continue
                
                # Check if class matches
                if pred['class'] == target['class']:
                    # Calculate IoU for defect positions within the signal
                    pred_start, pred_end = pred['defect_position']
                    target_start, target_end = target['defect_position']
                    
                    intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start))
                    union = max(pred_end, target_end) - min(pred_start, target_start)
                    
                    if union > 0:
                        iou = intersection / union
                        
                        # If IoU is above threshold, count as match
                        if iou > 0.5:
                            true_positives += 1
                            matched_targets.add(i)
                            matched = True
                            ious.append(iou)
                            break
            
            if not matched:
                false_positives += 1
        
        # Count unmatched targets as false negatives
        false_negatives += len(target_defects) - len(matched_targets)
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # Calculate mean IoU
    mean_iou = np.mean(ious) if ious else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    data_dir = "signal_dataset"
    sequences_file = os.path.join(data_dir, "signal_sequences.pt")
    save_dir = os.path.join(data_dir, "checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load dataset
    dataset = SignalSequenceDataset(sequences_file)
    print(f"Dataset size: {len(dataset)}")
    print(f"Label map: {dataset.label_map}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get signal length from first batch
    sample_batch = next(iter(train_loader))
    signal_length = sample_batch['signals'].shape[2]
    
    # Create model
    model = SignalSequenceDetector(
        signal_length=signal_length,
        d_model=128,
        num_classes=len(dataset.label_map),
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create optimizer with different learning rates for different components
    encoder_params = []
    transformer_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'signal_encoder' in name:
            encoder_params.append(param)
        elif 'sequence_transformer' in name or 'context_aggregator' in name or 'anomaly_detector' in name:
            transformer_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': transformer_params, 'lr': 5e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=0.01)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        device=device,
        save_dir=save_dir,
        log_interval=10
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(save_dir, 'training_history.png'))
    
    # Evaluate model
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device=device)
    
    # Print metrics
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training complete! Model saved to {save_dir}")
