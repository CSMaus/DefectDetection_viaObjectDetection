import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
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
from detection_models.direct_defect import DirectDefectModel
from detection_models.hybrid_binary import HybridBinaryModel

from defect_focused_dataset import get_defect_focused_dataloader


class CombinedDataset(Dataset):
    """Combines defective and clean sequences for balanced training"""
    def __init__(self, defective_dataset, clean_dataset):
        self.defective_dataset = defective_dataset
        self.clean_dataset = clean_dataset
        self.defective_len = len(defective_dataset)
        self.clean_len = len(clean_dataset)
        
        # Balance the datasets - use minimum length to ensure equal representation
        self.min_len = min(self.defective_len, self.clean_len)
        self.total_len = self.min_len * 2  # Equal amounts of each
        
        print(f"Balanced dataset: {self.min_len} defective + {self.min_len} clean = {self.total_len} total")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.min_len:
            # First half: defective sequences
            return self.defective_dataset[idx % self.defective_len]
        else:
            # Second half: clean sequences  
            clean_idx = (idx - self.min_len) % self.clean_len
            return self.clean_dataset[clean_idx]


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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.015)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # reduced gradient clipping
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
    
    # STEP 1: Load defective sequences to see how many we have
    print("Loading defective sequences...")
    train_loader_defects, val_loader_defects = get_defect_focused_dataloader(
        "json_data_0717",
        batch_size=8,
        seq_length=50,
        shuffle=True,
        validation_split=0.2,
        min_defects_per_sequence=1
    )
    
    num_defective = len(train_loader_defects.dataset) + len(val_loader_defects.dataset)
    print(f"Found {num_defective} defective sequences")
    
    # STEP 2: Create a custom dataset that loads ALL sequences (including clean ones)
    print(f"Now manually loading ALL sequences to select {num_defective} clean sequences for 1:1 ratio...")
    
    from defect_focused_dataset import DefectFocusedJsonSignalDataset
    import random
    
    # Create a modified version that loads ALL sequences by setting min_defects=0
    # But we need to manually modify the filtering logic
    
    class AllSequencesDataset(DefectFocusedJsonSignalDataset):
        """Modified dataset that loads ALL sequences, not just defective ones"""
        def _load_defect_sequences(self):
            """Override to load ALL sequences, both defective and clean"""
            total_sequences_created = 0
            total_sequences_with_defects = 0
            total_sequences_without_defects = 0
            
            print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")
            
            for json_file in self.json_files:
                json_path = os.path.join(self.json_dir, json_file)
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                for beam_key in data.keys():
                    beam_data = data[beam_key]
                    
                    # Sort scan keys by index
                    scans_keys_sorted = sorted(beam_data.keys(), key=lambda x: int(x.split('_')[0]))
                    
                    # Skip if not enough scans for a full sequence
                    if len(scans_keys_sorted) < self.seq_length:
                        continue
                    
                    # Extract all signals, labels, and defect positions for this beam
                    all_scans_for_beam = {}
                    all_lbls_for_beam = {}
                    all_defects_for_beam = {}
                    
                    scan_idx = 0
                    for scan_key in scans_keys_sorted:
                        scan_data = beam_data[scan_key]
                        all_scans_for_beam[str(scan_idx)] = scan_data
                        
                        # Extract label and defect position
                        if scan_key.split('_')[1] == "Health":
                            all_lbls_for_beam[str(scan_idx)] = 0
                            all_defects_for_beam[str(scan_idx)] = [None, None]
                        else:
                            all_lbls_for_beam[str(scan_idx)] = 1
                            try:
                                defect_range = scan_key.split('_')[2].split('-')
                                defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                                all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]
                            except:
                                all_defects_for_beam[str(scan_idx)] = [0.0, 1.0]
                        
                        scan_idx += 1
                    
                    # Create sequences from this beam
                    for start_idx in range(len(scans_keys_sorted) - self.seq_length + 1):
                        seq_signals = []
                        seq_labels = []
                        seq_positions = []
                        
                        for i in range(self.seq_length):
                            signal_idx = str(start_idx + i)
                            seq_signals.append(all_scans_for_beam[signal_idx])
                            seq_labels.append(all_lbls_for_beam[signal_idx])
                            
                            if all_defects_for_beam[signal_idx][0] is not None:
                                seq_positions.append(all_defects_for_beam[signal_idx])
                            else:
                                seq_positions.append([0.0, 0.0])
                        
                        # LOAD ALL SEQUENCES - both defective and clean
                        defect_count = sum(seq_labels)
                        total_sequences_created += 1
                        
                        if defect_count > 0:
                            total_sequences_with_defects += 1
                        else:
                            total_sequences_without_defects += 1
                        
                        # Convert to tensors
                        signal_tensor = torch.tensor(seq_signals, dtype=torch.float32)
                        label_tensor = torch.tensor(seq_labels, dtype=torch.float32)
                        position_tensor = torch.tensor(seq_positions, dtype=torch.float32)
                        
                        self.signal_sets.append(signal_tensor)
                        self.labels.append(label_tensor)
                        self.defect_positions.append(position_tensor)
            
            print(f"Total sequences created: {total_sequences_created}")
            print(f"Sequences with defects: {total_sequences_with_defects}")
            print(f"Sequences without defects: {total_sequences_without_defects}")
    
    # Load ALL sequences
    all_dataset = AllSequencesDataset("json_data_0717", seq_length=50, min_defects_per_sequence=0)
    
    # Separate defective and clean sequences
    defective_sequences = []
    clean_sequences = []
    
    for i in range(len(all_dataset)):
        signals, labels, positions = all_dataset[i]
        if labels.sum() > 0:  # Has defects
            defective_sequences.append((signals, labels, positions))
        else:  # Clean sequence
            clean_sequences.append((signals, labels, positions))
    
    print(f"Separated: {len(defective_sequences)} defective, {len(clean_sequences)} clean")
    
    # Select exactly the same number of clean sequences as defective
    num_to_select = len(defective_sequences)
    if len(clean_sequences) >= num_to_select:
        # Randomly select clean sequences to match defective count
        random.shuffle(clean_sequences)
        selected_clean = clean_sequences[:num_to_select]
        print(f"Selected {len(selected_clean)} clean sequences to match {num_to_select} defective sequences")
        
        # Combine for balanced dataset
        balanced_data = defective_sequences + selected_clean
        random.shuffle(balanced_data)  # Shuffle the combined data
        
        print(f"BALANCED DATASET: {len(balanced_data)} total sequences (1:1 ratio)")
        
        # Create balanced dataloaders
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        all_signals = torch.stack([item[0] for item in balanced_data])
        all_labels = torch.stack([item[1] for item in balanced_data])
        all_positions = torch.stack([item[2] for item in balanced_data])
        
        # Create dataset and split
        balanced_dataset = TensorDataset(all_signals, all_labels, all_positions)
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(balanced_dataset))
        val_size = len(balanced_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        print(f"Final balanced loaders: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
        
    else:
        print(f"ERROR: Not enough clean sequences! Need {num_to_select}, but only have {len(clean_sequences)}")
        return
    
    models = {
        # "ComplexONNX": ComplexDetectionModelONNX(signal_length=320)
        # "ComplexFix": ComplexDetectionModelFix(signal_length=320)
        # "NoiseRobust": NoiseRobustDetectionModel(signal_length=320)
        # "PatternEmbedding": PatternEmbeddingModel(signal_length=320)
        # "EnhancedPattern": EnhancedPatternModel(signal_length=320)
        # "DirectDefectModel": DirectDefectModel(signal_length=320, d_model=64, num_heads=16, num_layers=4,)dropout=0.5
        "HybridBinaryModel": HybridBinaryModel(signal_length=320)
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
            num_epochs=15, device=device, model_name=name, save_dir=main_models_dir
        )
        
        results[name] = best_acc
        all_histories[name] = history
        print(f"{name} Model Best Accuracy: {best_acc:.4f}")
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for name, acc in results.items():
        print(f"{name} Detection Model: {acc:.4f}")
    
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
