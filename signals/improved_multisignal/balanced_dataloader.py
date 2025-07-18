import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random


class BalancedJsonSignalDataset(Dataset):
    """
    Dataset that loads BOTH defective and clean sequences in 1:1 ratio
    Based on DefectFocusedJsonSignalDataset but includes clean sequences
    """
    def __init__(self, json_dir, seq_length=50):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.seq_length = seq_length
        self.signal_sets = []
        self.labels = []
        self.defect_positions = []
        
        # Load sequences and separate defective/clean
        self._load_balanced_sequences()
    
    def _load_balanced_sequences(self):
        """Load sequences and create balanced dataset with 1:1 defective:clean ratio"""
        defective_sequences = []
        clean_sequences = []
        
        total_sequences_created = 0
        total_beams = 0
        
        print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")
        
        for json_file in self.json_files:
            json_path = os.path.join(self.json_dir, json_file)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for beam_key in data.keys():
                beam_data = data[beam_key]
                total_beams += 1
                
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
                    
                    # Convert to tensors
                    signal_tensor = torch.tensor(seq_signals, dtype=torch.float32)
                    label_tensor = torch.tensor(seq_labels, dtype=torch.float32)
                    position_tensor = torch.tensor(seq_positions, dtype=torch.float32)
                    
                    total_sequences_created += 1
                    defect_count = sum(seq_labels)
                    
                    # Separate defective and clean sequences
                    if defect_count > 0:
                        defective_sequences.append((signal_tensor, label_tensor, position_tensor))
                    else:
                        clean_sequences.append((signal_tensor, label_tensor, position_tensor))
        
        print(f"Total beams: {total_beams}")
        print(f"Total sequences created: {total_sequences_created}")
        print(f"Sequences with defects: {len(defective_sequences)}")
        print(f"Sequences without defects: {len(clean_sequences)}")
        
        # CREATE BALANCED 1:1 RATIO
        num_defective = len(defective_sequences)
        num_clean = len(clean_sequences)
        
        if num_clean >= num_defective:
            # Randomly select clean sequences to match defective count
            random.shuffle(clean_sequences)
            selected_clean = clean_sequences[:num_defective]
            
            print(f"BALANCED SELECTION:")
            print(f"Defective sequences: {num_defective}")
            print(f"Selected clean sequences: {len(selected_clean)}")
            print(f"Total balanced sequences: {num_defective + len(selected_clean)}")
            
            # Combine and shuffle
            all_sequences = defective_sequences + selected_clean
            random.shuffle(all_sequences)
            
            # Extract tensors
            for signals, labels, positions in all_sequences:
                self.signal_sets.append(signals)
                self.labels.append(labels)
                self.defect_positions.append(positions)
                
        else:
            print(f"ERROR: Not enough clean sequences! Need {num_defective}, have {num_clean}")
            # Use all available sequences
            all_sequences = defective_sequences + clean_sequences
            for signals, labels, positions in all_sequences:
                self.signal_sets.append(signals)
                self.labels.append(labels)
                self.defect_positions.append(positions)
        
        print(f"Final dataset size: {len(self.signal_sets)} sequences")
        defect_ratio = sum(1 for labels in self.labels if labels.sum() > 0) / len(self.labels)
        print(f"Defect sequence ratio: {defect_ratio:.3f}")
    
    def __len__(self):
        return len(self.signal_sets)
    
    def __getitem__(self, idx):
        return self.signal_sets[idx], self.labels[idx], self.defect_positions[idx]


def get_balanced_dataloader(json_dir, batch_size=8, seq_length=50, shuffle=True, num_workers=4, validation_split=0.2):
    """
    Create balanced DataLoaders with 1:1 defective:clean ratio
    """
    # Create the balanced dataset
    full_dataset = BalancedJsonSignalDataset(json_dir, seq_length)
    
    # Calculate sizes for train and validation sets
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Balanced dataset split: {train_size} training samples, {val_size} validation samples")
    
    return train_loader, val_loader
