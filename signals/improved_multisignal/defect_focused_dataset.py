import torch
import numpy as np
import json
import os
import math
from torch.utils.data import Dataset, DataLoader, random_split


class DefectFocusedJsonSignalDataset(Dataset):
    """
    Dataset class for loading signal data from JSON files.
    ONLY includes sequences that contain at least one defect.
    This focuses training on defect localization rather than detection.
    """
    def __init__(self, json_dir, seq_length=50, min_defects_per_sequence=1):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.seq_length = seq_length
        self.min_defects_per_sequence = min_defects_per_sequence
        self.signal_sets = []
        self.labels = []
        self.defect_positions = []
        
        # Load all JSON files and extract sequences with defects
        self._load_defect_sequences()
        
        print(f"Loaded {len(self.signal_sets)} DEFECT-CONTAINING sequences from {len(self.json_files)} JSON files")
        print(f"Each sequence contains {self.seq_length} signals")
        print(f"Minimum defects per sequence: {self.min_defects_per_sequence}")
    
    def _load_defect_sequences(self):
        """Load ONLY sequences that contain defects"""
        total_beams = 0
        total_sequences = 0
        total_sequences_with_defects = 0
        total_sequences_skipped = 0
        
        print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")
        
        # Process each JSON file
        for json_file in self.json_files:
            file_path = os.path.join(self.json_dir, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process each beam
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
                                defect_part = scan_key.split('_')[2]
                                # Handle both formats: "0-14" and "0_14" 
                                if '-' in defect_part:
                                    # Handle negative start: "-5-14" vs "0-14"
                                    if defect_part.startswith('-'):
                                        # Negative start: "-5-14"
                                        remaining = defect_part[1:]  # Remove first '-'
                                        if '-' in remaining:
                                            parts = remaining.split('-', 1)
                                            defect_start = -float(parts[0])
                                            defect_end = float(parts[1])
                                        else:
                                            defect_start = -float(remaining)
                                            defect_end = defect_start + 1.0
                                    else:
                                        # Positive start: "0-14"
                                        defect_range = defect_part.split('-')
                                        defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                                else:
                                    # Handle underscore format if it exists in scan keys
                                    if len(scan_key.split('_')) >= 4:
                                        defect_start = float(scan_key.split('_')[2])
                                        defect_end = float(scan_key.split('_')[3])
                                    else:
                                        defect_start = float(defect_part)
                                        defect_end = defect_start + 1.0
                                
                                all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]
                            except:
                                all_defects_for_beam[str(scan_idx)] = [0.0, 0.0]
                        
                        scan_idx += 1
                    
                    # Create sequences from this beam
                    num_of_seqs_for_beam = math.ceil(len(scans_keys_sorted) / self.seq_length)
                    
                    for i in range(num_of_seqs_for_beam):
                        sequence = []
                        seq_labels = []
                        seq_defects = []
                        
                        # Determine start and end indices for this sequence
                        if i < num_of_seqs_for_beam - 1:
                            start_idx = i * self.seq_length
                            end_idx = start_idx + self.seq_length
                        else:
                            start_idx = len(scans_keys_sorted) - self.seq_length
                            end_idx = len(scans_keys_sorted)
                        
                        if start_idx < 0:
                            continue
                        
                        # Extract signals for this sequence
                        for j in range(start_idx, end_idx):
                            try:
                                scan_data = all_scans_for_beam[str(j)]
                                
                                # Convert scan data to numpy array
                                if isinstance(scan_data, list):
                                    signal = np.array(scan_data, dtype=np.float32)
                                elif isinstance(scan_data, dict) and 'signal' in scan_data:
                                    signal = np.array(scan_data['signal'], dtype=np.float32)
                                else:
                                    signal = np.array(scan_data, dtype=np.float32)
                                
                                sequence.append(signal)
                                seq_labels.append(all_lbls_for_beam[str(j)])
                                seq_defects.append(all_defects_for_beam[str(j)])
                            except Exception as e:
                                print(f"Error processing scan {j} in beam {beam_key}: {e}")
                                continue
                        
                        # Skip if sequence doesn't have exactly seq_length signals
                        if len(sequence) != self.seq_length:
                            continue
                        
                        # Ensure all signals have the same length
                        signal_length = len(sequence[0])
                        valid = True
                        for signal in sequence:
                            if len(signal) != signal_length:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        total_sequences += 1
                        
                        # *** KEY CHANGE: Only include sequences with defects ***
                        defect_count = sum(seq_labels)
                        if defect_count < self.min_defects_per_sequence:
                            total_sequences_skipped += 1
                            continue  # Skip sequences without enough defects
                        
                        # Format defects
                        formatted_defects = []
                        for defect in seq_defects:
                            if defect[0] is None:
                                formatted_defects.append([0.0, 0.0])
                            else:
                                formatted_defects.append([float(defect[0]), float(defect[1])])
                        
                        # Add to dataset (only sequences with defects)
                        self.signal_sets.append(np.array(sequence, dtype=np.float32))
                        self.labels.append(np.array(seq_labels, dtype=np.float32))
                        self.defect_positions.append(np.array(formatted_defects, dtype=np.float32))
                        total_sequences_with_defects += 1
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Total beams: {total_beams}")
        print(f"Total sequences created: {total_sequences}")
        print(f"Sequences with defects (kept): {total_sequences_with_defects}")
        print(f"Sequences without defects (skipped): {total_sequences_skipped}")
        print(f"Defect sequence ratio: {total_sequences_with_defects/total_sequences:.3f}")
    
    def __len__(self):
        return len(self.signal_sets)
    
    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        defect_positions = torch.tensor(self.defect_positions[idx], dtype=torch.float32)
        return signals, labels, defect_positions


def get_defect_focused_dataloader(json_dir, batch_size=4, seq_length=50, shuffle=True, num_workers=4, validation_split=0.2, min_defects_per_sequence=1):
    """
    Create DataLoaders for training and validation with defect-focused sequences
    
    Args:
        json_dir: Directory containing JSON files
        batch_size: Batch size for training
        seq_length: Number of signals per sequence
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
        min_defects_per_sequence: Minimum number of defects required per sequence
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create the defect-focused dataset
    full_dataset = DefectFocusedJsonSignalDataset(json_dir, seq_length, min_defects_per_sequence)
    
    # Calculate sizes for train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Defect-focused dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
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
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
