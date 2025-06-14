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
import argparse
import sys

from dataset_preparation import SignalSequenceDataset
from two_stage_model import TwoStageDefectDetector


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and complex target structures.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Collated batch with consistent structure
    """
    # Extract signals and targets
    signals = [item['signals'] for item in batch]
    targets = [item['targets'] for item in batch]
    
    # Get other metadata
    file_names = [item['file_name'] for item in batch]
    scan_keys = [item['scan_key'] for item in batch]
    
    # Stack signals (they should all have the same shape after padding)
    signals = torch.stack(signals)
    
    return {
        'signals': signals,
        'targets': targets,
        'file_name': file_names,
        'scan_key': scan_keys
    }


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=20,
    device='cpu',
    save_dir='checkpoints',
    log_interval=10,
    patience=5
):
    """
    Train the model with early stopping and gradient accumulation.
    
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
        patience: Patience for early stopping
        
    Returns:
        dict: Training history
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cls_loss': [],
        'train_position_loss': [],
        'train_uncertainty_loss': [],
        'val_cls_loss': [],
        'val_position_loss': [],
        'val_uncertainty_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_position_loss = 0.0
        train_uncertainty_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            signals = batch['signals'].to(device)
            targets = batch['targets']
            
            optimizer.zero_grad()
            loss, loss_dict = model(signals, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += loss_dict['cls_loss'].item()
            train_position_loss += loss_dict['position_loss'].item()
            train_uncertainty_loss += loss_dict['uncertainty_loss'].item()
            
            if (batch_idx + 1) % log_interval == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Cls: {loss_dict['cls_loss'].item():.4f}, "
                      f"Pos: {loss_dict['position_loss'].item():.4f}, "
                      f"Uncert: {loss_dict['uncertainty_loss'].item():.4f}")
        
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_position_loss /= len(train_loader)
        train_uncertainty_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_position_loss = 0.0
        val_uncertainty_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                signals = batch['signals'].to(device)
                targets = batch['targets']
                
                loss, loss_dict = model(signals, targets)
                
                val_loss += loss.item()
                val_cls_loss += loss_dict['cls_loss'].item()
                val_position_loss += loss_dict['position_loss'].item()
                val_uncertainty_loss += loss_dict['uncertainty_loss'].item()
        
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_position_loss /= len(val_loader)
        val_uncertainty_loss /= len(val_loader)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_position_loss'].append(train_position_loss)
        history['train_uncertainty_loss'].append(train_uncertainty_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_position_loss'].append(val_position_loss)
        history['val_uncertainty_loss'].append(val_uncertainty_loss)
        history['learning_rate'].append(current_lr)
        
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Cls: {train_cls_loss:.4f}, Val Cls: {val_cls_loss:.4f}")
        print(f"  Train Pos: {train_position_loss:.4f}, Val Pos: {val_position_loss:.4f}")
        print(f"  Train Uncert: {train_uncertainty_loss:.4f}, Val Uncert: {val_uncertainty_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
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
        
        torch.save(checkpoint, os.path.join(save_dir, 'last_model.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            patience_counter = 0
            print("  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    return history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_position_loss = 0.0
    total_uncertainty_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            signals = batch['signals'].to(device)
            targets = batch['targets']
            
            loss, loss_dict = model(signals, targets)
            
            total_loss += loss.item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_position_loss += loss_dict['position_loss'].item()
            total_uncertainty_loss += loss_dict['uncertainty_loss'].item()
            
            # Get predictions
            preds = model.predict(signals)
            all_preds.extend(preds)
            
            # Process targets for metrics calculation
            batch_targets = []
            for b in range(len(targets)):
                sequence_targets = []
                for i, target in enumerate(targets[b]):
                    target_defects = []
                    if target['label'] > 0:  # If it's a defect (not "Health")
                        target_defects.append({
                            'position': i,
                            'class': target['label'],
                            'defect_position': target['defect_position'].cpu().numpy() if torch.is_tensor(target['defect_position']) else target['defect_position']
                        })
                    sequence_targets.extend(target_defects)
                batch_targets.append(sequence_targets)
            all_targets.extend(batch_targets)
    
    # Calculate average losses
    avg_loss = total_loss / len(test_loader)
    avg_cls_loss = total_cls_loss / len(test_loader)
    avg_position_loss = total_position_loss / len(test_loader)
    avg_uncertainty_loss = total_uncertainty_loss / len(test_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Add losses to metrics
    metrics['loss'] = avg_loss
    metrics['cls_loss'] = avg_cls_loss
    metrics['position_loss'] = avg_position_loss
    metrics['uncertainty_loss'] = avg_uncertainty_loss
    
    return metrics


def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        iou_threshold: IoU threshold for considering a prediction correct
        
    Returns:
        dict: Metrics
    """
    total_pred = sum(len(p) for p in predictions)
    total_target = sum(len(t) for t in targets)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    position_errors = []
    
    for batch_idx in range(len(predictions)):
        batch_preds = predictions[batch_idx]
        batch_targets = targets[batch_idx] if batch_idx < len(targets) else []
        
        # Match predictions to targets
        matched_targets = set()
        
        for pred in batch_preds:
            pred_pos = pred['position']
            pred_defect_pos = pred['defect_position']
            
            best_iou = 0
            best_target_idx = -1
            
            for target_idx, target in enumerate(batch_targets):
                if target_idx in matched_targets:
                    continue
                
                target_pos = target['position']
                target_defect_pos = target['defect_position']
                
                # Check if prediction and target are for the same signal
                if pred_pos == target_pos:
                    # Calculate IoU for defect positions
                    pred_start, pred_end = pred_defect_pos
                    target_start, target_end = target_defect_pos
                    
                    intersection = max(0, min(pred_end, target_end) - max(pred_start, target_start))
                    union = max(pred_end, target_end) - min(pred_start, target_start)
                    
                    if union > 0:
                        iou = intersection / union
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = target_idx
            
            if best_iou > iou_threshold:
                true_positives += 1
                matched_targets.add(best_target_idx)
                
                # Calculate position error
                target = batch_targets[best_target_idx]
                pred_start, pred_end = pred_defect_pos
                target_start, target_end = target['defect_position']
                
                start_error = abs(pred_start - target_start)
                end_error = abs(pred_end - target_end)
                position_errors.append((start_error + end_error) / 2)
            else:
                false_positives += 1
        
        # Count unmatched targets as false negatives
        false_negatives += len(batch_targets) - len(matched_targets)
    
    # Calculate metrics
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
    
    mean_position_error = np.mean(position_errors) if position_errors else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_position_error': mean_position_error,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history dict
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot classification loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_cls_loss'], label='Train Cls Loss')
    plt.plot(history['val_cls_loss'], label='Val Cls Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot position loss
    plt.subplot(2, 2, 3)
    plt.plot(history['train_position_loss'], label='Train Pos Loss')
    plt.plot(history['val_position_loss'], label='Val Pos Loss')
    plt.title('Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot uncertainty loss
    plt.subplot(2, 2, 4)
    plt.plot(history['train_uncertainty_loss'], label='Train Uncert Loss')
    plt.plot(history['val_uncertainty_loss'], label='Val Uncert Loss')
    plt.title('Uncertainty Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"two_stage_model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    dataset = SignalSequenceDataset(args.data_path)
    print(f"Dataset size: {len(dataset)}")
    print(f"Label map: {dataset.label_map}")
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Get a sample batch to determine signal length
    sample_batch = next(iter(train_loader))
    signal_length = sample_batch['signals'].shape[2]
    
    model = TwoStageDefectDetector(
        signal_length=signal_length,
        d_model=args.d_model,
        num_classes=len(dataset.label_map)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create optimizer with different learning rates for different parts of the model
    encoder_params = []
    transformer_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'signal_encoder' in name:
                encoder_params.append(param)
            elif 'sequence_transformer' in name:
                transformer_params.append(param)
            else:
                head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr},
        {'params': transformer_params, 'lr': args.lr * 2},
        {'params': head_params, 'lr': args.lr * 3}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=save_dir,
        log_interval=10,
        patience=args.patience
    )
    
    plot_training_history(history, save_path=os.path.join(save_dir, 'training_history.png'))
    
    best_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device=device)
    
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config = {
            'data_path': 'signal_dataset_38/signal_sequences.pkl',
            'save_dir': 'two_stage_models',
            'batch_size': 16,  # todo: set 16
            'epochs': 10,
            'lr': 1e-4,
            'patience': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'd_model': 128
        }

        class Args:
            pass
        c_args = Args()
        for key, value in config.items():
            setattr(c_args, key, value)


    else:
        parser = argparse.ArgumentParser(description="Train a two-stage defect detection model")

        parser.add_argument("--data_path", type=str, default="signal_dataset_38/signal_sequences.pkl",
                            help="Path to the dataset file")
        parser.add_argument("--save_dir", type=str, default="models",
                            help="Directory to save models")
        parser.add_argument("--batch_size", type=int, default=8,
                            help="Batch size for training")
        parser.add_argument("--epochs", type=int, default=30,
                            help="Number of epochs to train")
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
        parser.add_argument("--patience", type=int, default=5,
                            help="Patience for early stopping")
        parser.add_argument("--d_model", type=int, default=128,
                            help="Model dimension")
        parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use (cuda or cpu)")
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed")
        c_args = parser.parse_args()
        torch.manual_seed(c_args.seed)
        np.random.seed(c_args.seed)
    
    main(c_args)
