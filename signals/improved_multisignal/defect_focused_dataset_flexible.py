import torch
import numpy as np
import json
import os
import math
import re
from torch.utils.data import Dataset, DataLoader, random_split


class FlexibleDefectFocusedJsonSignalDataset(Dataset):
    """
    Dataset class for loading signal data from JSON files.
    ONLY includes sequences that contain at least one defect.
    This focuses training on defect localization rather than detection.
    
    Supports both old and new file naming conventions:
    - Old: WOT_D33-D36_01_Ch-0_D0-14.json
    - New: WOT_D33-D36_01_Ch-0-S350_400-D0_14.json
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
    
    def _extract_defect_info_from_scan_key(self, scan_key):
        """
        Extract defect information from scan key, supporting both old and new formats.
        FIXED: Now properly handles negative defect start positions.
        
        Old format: "0_Health" or "0_Defect_0-14" or "0_Defect_-5-14" (negative start)
        New format: "0_Health" or "0_Defect_0_14" or "0_Defect_-5_14" (negative start)
        
        Returns:
            tuple: (is_defect, defect_start, defect_end)
        """
        parts = scan_key.split('_')
        
        if len(parts) < 2:
            return False, None, None
        
        # Check if it's a health scan
        if parts[1] == "Health":
            return False, None, None
        
        # It's a defect scan - extract defect range
        if len(parts) >= 3:
            try:
                # Try different formats for defect range
                defect_part = parts[2]
                
                # Handle format like "0-14" or "-5-14" (with dash)
                if '-' in defect_part:
                    # Need to be careful with negative numbers
                    # Split by '-' but handle negative start values
                    if defect_part.startswith('-'):
                        # Negative start: "-5-14" -> ["-5", "14"]
                        remaining = defect_part[1:]  # Remove first '-'
                        if '-' in remaining:
                            parts_split = remaining.split('-', 1)
                            defect_start = -float(parts_split[0])  # Make it negative
                            defect_end = float(parts_split[1])
                        else:
                            # Just negative number: "-5" -> assume small range
                            defect_start = -float(remaining)
                            defect_end = defect_start + 1.0
                    else:
                        # Positive start: "0-14" -> ["0", "14"]
                        defect_range = defect_part.split('-')
                        if len(defect_range) == 2:
                            defect_start = float(defect_range[0])
                            defect_end = float(defect_range[1])
                        else:
                            # Single positive number
                            defect_start = float(defect_part)
                            defect_end = defect_start + 1.0
                    
                    return True, defect_start, defect_end
                
                # Handle format like "0_14" or "-5_14" (parts[2] and parts[3])
                if len(parts) >= 4:
                    defect_start = float(parts[2])  # Can be negative
                    defect_end = float(parts[3])    # Always positive
                    return True, defect_start, defect_end
                
                # Single number case (can be negative)
                defect_pos = float(defect_part)
                return True, defect_pos, defect_pos + 1.0
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse defect range from scan key '{scan_key}': {e}")
                return True, 0.0, 1.0  # Default small defect
        
        # Default case for defect without clear range
        return True, 0.0, 1.0
    
    def _detect_file_format(self, filename):
        """
        Detect whether the file uses old or new naming convention.
        
        Old: WOT_D33-D36_01_Ch-0_D0-14.json
        New: WOT_D33-D36_01_Ch-0-S350_400-D0_14.json
        
        Returns:
            str: 'old' or 'new'
        """
        # Look for the S pattern (S followed by numbers, underscore, numbers)
        s_pattern = r'-S\d+_\d+-'
        if re.search(s_pattern, filename):
            return 'new'
        else:
            return 'old'
    
    def _extract_defect_from_filename(self, filename):
        """
        Extract defect information from filename for additional validation.
        FIXED: Now properly handles negative defect start positions.
        
        Args:
            filename: JSON filename
            
        Returns:
            tuple: (has_defect_in_name, defect_start, defect_end) or (False, None, None)
        """
        try:
            file_format = self._detect_file_format(filename)
            
            if file_format == 'new':
                # New format: WOT_D33-D36_01_Ch-0-S350_400-D0_14.json or WOT_D33-D36_01_Ch-0-S350_400-D-5_14.json
                # Look for the D part after S part - FIXED to handle negative numbers
                d_pattern = r'-D(-?\d+)_(\d+)\.json$'
                match = re.search(d_pattern, filename)
                if match:
                    defect_start = float(match.group(1))  # Can be negative
                    defect_end = float(match.group(2))    # Always positive
                    return True, defect_start, defect_end
            else:
                # Old format: WOT_D33-D36_01_Ch-0_D0-14.json or WOT_D33-D36_01_Ch-0_D-5-14.json
                # Look for the D part at the end - FIXED to handle negative numbers
                d_pattern = r'_D(-?\d+)-(\d+)\.json$'
                match = re.search(d_pattern, filename)
                if match:
                    defect_start = float(match.group(1))  # Can be negative
                    defect_end = float(match.group(2))    # Always positive
                    return True, defect_start, defect_end
            
            return False, None, None
            
        except Exception as e:
            print(f"Warning: Could not parse defect info from filename '{filename}': {e}")
            return False, None, None
    
    def _load_defect_sequences(self):
        """Load ONLY sequences that contain defects"""
        total_beams = 0
        total_sequences = 0
        total_sequences_with_defects = 0
        total_sequences_skipped = 0
        
        print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")
        
        # Analyze file formats
        old_format_count = 0
        new_format_count = 0
        negative_defect_count = 0
        
        for json_file in self.json_files:
            if self._detect_file_format(json_file) == 'new':
                new_format_count += 1
            else:
                old_format_count += 1
            
            # Check for negative defects in filename
            has_defect, start, end = self._extract_defect_from_filename(json_file)
            if has_defect and start is not None and start < 0:
                negative_defect_count += 1
        
        print(f"File format analysis: {old_format_count} old format, {new_format_count} new format")
        print(f"Files with negative defect start positions: {negative_defect_count}")
        
        # Process each JSON file
        for json_file in self.json_files:
            file_path = os.path.join(self.json_dir, json_file)
            file_format = self._detect_file_format(json_file)
            
            # Extract defect info from filename for validation
            filename_has_defect, filename_defect_start, filename_defect_end = self._extract_defect_from_filename(json_file)
            
            if filename_has_defect and filename_defect_start is not None:
                print(f"Processing {json_file}: defect range [{filename_defect_start}, {filename_defect_end}]")
            
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
                        
                        # Extract label and defect position using flexible method
                        is_defect, defect_start, defect_end = self._extract_defect_info_from_scan_key(scan_key)
                        
                        if not is_defect:
                            all_lbls_for_beam[str(scan_idx)] = 0
                            all_defects_for_beam[str(scan_idx)] = [None, None]
                        else:
                            all_lbls_for_beam[str(scan_idx)] = 1
                            
                            # Use filename defect info as fallback if scan key parsing failed
                            if defect_start is None and filename_has_defect:
                                defect_start = filename_defect_start
                                defect_end = filename_defect_end
                            elif defect_start is None:
                                defect_start = 0.0
                                defect_end = 1.0
                            
                            all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]
                        
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


def get_flexible_defect_focused_dataloader(json_dir, batch_size=4, seq_length=50, shuffle=True, num_workers=4, validation_split=0.2, min_defects_per_sequence=1):
    """
    Create DataLoaders for training and validation with defect-focused sequences.
    Supports both old and new JSON file naming conventions.
    
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
    # Create the flexible defect-focused dataset
    full_dataset = FlexibleDefectFocusedJsonSignalDataset(json_dir, seq_length, min_defects_per_sequence)
    
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
    
    print(f"Flexible defect-focused dataset split: {train_size} training samples, {val_size} validation samples")
    
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


# Backward compatibility - keep the original function name
def get_defect_focused_dataloader(json_dir, batch_size=4, seq_length=50, shuffle=True, num_workers=4, validation_split=0.2, min_defects_per_sequence=1):
    """Backward compatibility wrapper"""
    return get_flexible_defect_focused_dataloader(json_dir, batch_size, seq_length, shuffle, num_workers, validation_split, min_defects_per_sequence)
