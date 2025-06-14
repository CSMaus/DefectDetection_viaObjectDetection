import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader


class JsonSignalDataset(Dataset):
    """
    Dataset class for loading signal data from JSON files.
    Maintains the same structure as the original SignalDataset in training_01.py.
    Only includes sequences with defects.
    Creates multiple sequences from beams with more than seq_length signals.
    """
    def __init__(self, json_dir, seq_length=50):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.seq_length = seq_length
        self.signal_sets = []
        self.labels = []
        self.defect_positions = []
        
        # Load all JSON files and extract sequences with defects
        self._load_all_json_files()
        
        print(f"Loaded {len(self.signal_sets)} sequences with defects from {len(self.json_files)} JSON files")
        print(f"Each sequence contains {self.seq_length} signals")
    
    def _load_all_json_files(self):
        """Load ALL JSON files and combine their sequences into a single dataset"""
        total_beams = 0
        beams_with_defects = 0
        total_sequences = 0
        
        print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")
        
        # Process each JSON file (equivalent to a folder in the original implementation)
        for json_file in self.json_files:
            file_path = os.path.join(self.json_dir, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process each beam (equivalent to a beam folder in the original implementation)
                for beam_key, beam_data in data.items():
                    total_beams += 1
                    
                    # Check if this beam has any defects
                    has_defects = False
                    for signal_key in beam_data.keys():
                        if '_' in signal_key and signal_key.split('_')[1] != 'Health':
                            has_defects = True
                            break
                    
                    # Skip beams without defects
                    if not has_defects:
                        continue
                    
                    beams_with_defects += 1
                    
                    # Sort signal files by scan index (equivalent to sorting txt files in the original implementation)
                    signal_files = sorted(list(beam_data.keys()), 
                                         key=lambda x: int(round(float(x.split('_')[0]))) if '_' in x and x.split('_')[0].replace('.', '', 1).isdigit() else 0)
                    
                    if not signal_files:
                        continue
                    
                    # Extract all signals for this beam
                    all_signals = []
                    all_labels = []
                    all_defect_positions = []
                    
                    for filename in signal_files:
                        # Extract signal data (equivalent to loading from txt file)
                        signal_data = beam_data[filename]
                        
                        # Handle different signal data formats
                        if isinstance(signal_data, list):
                            signal = np.array(signal_data, dtype=np.float32)
                        elif isinstance(signal_data, dict) and 'signal' in signal_data:
                            signal = np.array(signal_data['signal'], dtype=np.float32)
                        else:
                            try:
                                signal = np.array(signal_data, dtype=np.float32)
                            except:
                                print(f"Could not extract signal from {filename} in {json_file}, beam {beam_key}")
                                continue
                        
                        all_signals.append(signal)
                        
                        # Process label and defect position (exactly as in the original implementation)
                        defect_name = filename.split('_')[1]
                        if defect_name == 'Health':
                            all_labels.append(0.0)
                            all_defect_positions.append([0.0, 0.0])
                        else:
                            try:
                                defect_range = filename.split('_')[2].split('-')
                                defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                                all_labels.append(1.0)
                                all_defect_positions.append([defect_start, defect_end])
                            except Exception as e:
                                print(f"Error processing defect position for {filename}: {e}")
                                all_labels.append(1.0)
                                all_defect_positions.append([0.0, 0.0])
                    
                    # Skip if no valid signals were found
                    if not all_signals:
                        continue
                    
                    # Create sequences from this beam
                    num_signals = len(all_signals)
                    
                    if num_signals <= self.seq_length:
                        # If we have fewer signals than seq_length, pad with zeros
                        signals = all_signals.copy()
                        labels = all_labels.copy()
                        defect_positions = all_defect_positions.copy()
                        
                        # Pad with zeros
                        pad_length = self.seq_length - len(signals)
                        signal_length = len(signals[0])
                        signals.extend([np.zeros(signal_length, dtype=np.float32) for _ in range(pad_length)])
                        labels.extend([0.0 for _ in range(pad_length)])
                        defect_positions.extend([[0.0, 0.0] for _ in range(pad_length)])
                        
                        # Add to dataset
                        self.signal_sets.append(np.array(signals, dtype=np.float32))
                        self.labels.append(np.array(labels, dtype=np.float32))
                        self.defect_positions.append(np.array(defect_positions, dtype=np.float32))
                        total_sequences += 1
                    else:
                        # If we have more signals than seq_length, create multiple overlapping sequences
                        # Calculate step size to ensure all signals are used with some overlap
                        total_sequences_needed = max(1, int(np.ceil(num_signals / (self.seq_length / 2))))
                        step_size = max(1, int(np.floor((num_signals - self.seq_length) / (total_sequences_needed - 1)))) if total_sequences_needed > 1 else self.seq_length
                        
                        # Create sequences with the calculated step size
                        for start_idx in range(0, num_signals - self.seq_length + 1, step_size):
                            end_idx = start_idx + self.seq_length
                            
                            # Extract sequence
                            signals = all_signals[start_idx:end_idx]
                            labels = all_labels[start_idx:end_idx]
                            defect_positions = all_defect_positions[start_idx:end_idx]
                            
                            # Add to dataset
                            self.signal_sets.append(np.array(signals, dtype=np.float32))
                            self.labels.append(np.array(labels, dtype=np.float32))
                            self.defect_positions.append(np.array(defect_positions, dtype=np.float32))
                            total_sequences += 1
                        
                        # Add the last sequence if needed
                        if (num_signals - self.seq_length) % step_size != 0:
                            # Extract the last sequence
                            signals = all_signals[-self.seq_length:]
                            labels = all_labels[-self.seq_length:]
                            defect_positions = all_defect_positions[-self.seq_length:]
                            
                            # Add to dataset
                            self.signal_sets.append(np.array(signals, dtype=np.float32))
                            self.labels.append(np.array(labels, dtype=np.float32))
                            self.defect_positions.append(np.array(defect_positions, dtype=np.float32))
                            total_sequences += 1
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Total beams: {total_beams}, Beams with defects: {beams_with_defects}, Total sequences: {total_sequences}")
    
    def __len__(self):
        return len(self.signal_sets)
    
    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        defect_positions = torch.tensor(self.defect_positions[idx], dtype=torch.float32)
        return signals, labels, defect_positions


def get_dataloader(json_dir, batch_size=4, seq_length=50, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the JsonSignalDataset
    
    Args:
        json_dir: Directory containing JSON files
        batch_size: Batch size for training
        seq_length: Number of signals per sequence
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader object
    """
    dataset = JsonSignalDataset(json_dir, seq_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
