import os
import json
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SignalSequencePreparation:
    """
    Class for preparing signal sequences from JSON files.
    Instead of creating images, this class creates sequences of signals.
    """
    def __init__(self, ds_folder, output_folder, seq_length=50):
        """
        Initialize the signal sequence preparation.
        
        Args:
            ds_folder (str): Path to the folder containing JSON files
            output_folder (str): Path to save the processed dataset
            seq_length (int): Length of each sequence
        """
        self.ds_folder = ds_folder
        self.output_folder = output_folder
        self.seq_length = seq_length
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        self.number_false_signals = 0
        self.all_sequences = {}
        self.all_annotations = {}
        
    def get_datafile_sequences(self, json_filename):
        """
        Extract sequences and annotations from a JSON file.
        
        Args:
            json_filename (str): Name of the JSON file
            
        Returns:
            tuple: (sequences, annotations, beam_limits)
        """
        file_path = os.path.join(self.ds_folder, json_filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        sequence_name = os.path.splitext(json_filename)[0]
        beams = sorted(data.keys(), key=lambda beam_i: float(beam_i.split('_')[1]))
        beam_start = float(beams[0].split('_')[1])
        beam_end = float(beams[-1].split('_')[1])

        sequence = {}
        annotation_for_seq = {}

        for beam in beams:
            beam_idx = float(beam.split('_')[1])
            scan_dict = data[beam]
            scan_files = list(scan_dict.keys())

            for scan_file in scan_files:
                scan_key = scan_file.split('_')[0]
                if scan_key not in sequence:
                    sequence[scan_key] = []

                signal = scan_dict[scan_file]
                sequence[scan_key].append(signal)

                # Process annotations
                if scan_file.split('_')[1] == 'Health':
                    if scan_key not in annotation_for_seq:
                        annotation_for_seq[scan_key] = []
                else:
                    try:
                        defect_start_end = scan_file.split('_')[-1].split('-')
                        defect_start = float(defect_start_end[0])
                        defect_end = float(defect_start_end[1])

                        if scan_key not in annotation_for_seq or len(annotation_for_seq[scan_key]) == 0:
                            lbl = scan_file.split('_')[1]
                            annotation_for_seq[scan_key] = [{
                                "bbox": [beam_idx, beam_idx, defect_start, defect_end],
                                "label": lbl
                            }]
                        else:
                            condition = bool(annotation_for_seq[scan_key][-1]["bbox"][2] == defect_start
                                            and annotation_for_seq[scan_key][-1]["bbox"][3] == defect_end
                                            and annotation_for_seq[scan_key][-1]["bbox"][1] == beam_idx - 1)
                            
                            if condition:
                                annotation_for_seq[scan_key][-1]["bbox"][1] += 1
                            else:
                                # New defect
                                annotation_for_seq[scan_key].append({
                                    "bbox": [beam_idx, beam_idx,
                                            defect_start, defect_end],
                                    "label": scan_file.split('_')[1]
                                })
                    except Exception as ex:
                        print(f"Error: {ex} in {sequence_name}, {beam}, {scan_file}")
                        continue

            # Convert to numpy arrays at the end of processing each beam
            if beam == beams[-1]:
                for scan_key in sequence:
                    sequence[scan_key] = np.array(sequence[scan_key])

        # Sort by scan key
        annotation_for_seq_sorted = collections.OrderedDict(
            sorted(annotation_for_seq.items(), key=lambda x: int(x[0])))
        sequence_sorted = collections.OrderedDict(
            sorted(sequence.items(), key=lambda x: int(x[0])))
            
        return sequence_sorted, annotation_for_seq_sorted, (beam_start, beam_end)
    
    def normalize_annotations(self, annotations, beam_lims):
        """
        Normalize annotations for signal sequences.
        
        Args:
            annotations (dict): Annotations dictionary
            beam_lims (tuple): Beam limits (start, end)
            
        Returns:
            dict: Normalized annotations
        """
        beam_start, beam_end = beam_lims
        beam_len = beam_end - beam_start
        
        normalized_annotations = {}
        
        for scan_key, annots in annotations.items():
            normalized_annotations[scan_key] = []
            
            for defect in annots:
                # Normalize beam positions to [0, 1]
                beam_pos_start = (defect["bbox"][0] - beam_start) / beam_len
                beam_pos_end = (defect["bbox"][1] - beam_start) / beam_len
                
                # Keep defect positions as is (they should already be in [0, 1] range)
                # These represent the start and end positions within the 1D signal
                defect_start = defect["bbox"][2]
                defect_end = defect["bbox"][3]
                
                normalized_annotations[scan_key].append({
                    "bbox": [beam_pos_start, beam_pos_end, defect_start, defect_end],
                    "label": defect["label"]
                })
                
        return normalized_annotations
    
    def create_signal_sequences(self):
        """
        Process all JSON files and create signal sequences.
        
        Returns:
            tuple: (sequences, annotations)
        """
        json_files = [f for f in os.listdir(self.ds_folder) if f.endswith('.json')]
        
        for json_file in tqdm(json_files, desc="Processing JSON files", unit="file"):
            base_name = os.path.splitext(json_file)[0]
            
            seq, ann, blims = self.get_datafile_sequences(json_file)
            
            # Normalize annotations
            ann = self.normalize_annotations(ann, blims)
            
            self.all_sequences[base_name] = seq
            self.all_annotations[base_name] = ann
        
        beam_sequences = self.create_beam_sequences()
        
        with open(os.path.join(self.output_folder, "signal_annotations.json"), "w") as f:
            json.dump(self.all_annotations, f, indent=2)
            
        with open(os.path.join(self.output_folder, "signal_sequences.pkl"), 'wb') as f:
            import pickle
            pickle.dump(beam_sequences, f)
        
        print(f"Number of signals with values only zeroes: {self.number_false_signals}")
        print(f"Created {len(beam_sequences)} signal sequences")
        
        return beam_sequences, self.all_annotations
    
    def create_beam_sequences(self):
        """
        Create sequences of signals for each beam, ensuring all have the same length.
        
        Returns:
            list: List of sequences
        """
        beam_sequences = []
        
        # Process each file
        for file_name, sequences in self.all_sequences.items():
            # Process each beam (scan key)
            for scan_key, signals in sequences.items():
                # Check if signals are valid
                if np.all(signals == 0):
                    self.number_false_signals += 1
                    continue
                
                annotations = self.all_annotations[file_name].get(scan_key, [])
                
                num_signals = len(signals)
                
                if num_signals < self.seq_length:
                    # Pad with zeros to reach seq_length
                    padded_signals = np.zeros((self.seq_length, signals.shape[1]), dtype=signals.dtype)
                    padded_signals[:num_signals] = signals
                    
                    beam_sequences.append({
                        'file_name': file_name,
                        'scan_key': scan_key,
                        'signals': padded_signals,
                        'annotations': annotations,
                        'original_length': num_signals  # Store original length for reference
                    })
                elif num_signals == self.seq_length:
                    # Use as is
                    beam_sequences.append({
                        'file_name': file_name,
                        'scan_key': scan_key,
                        'signals': signals,
                        'annotations': annotations
                    })
                else:
                    # Create overlapping sequences to use all signals
                    # Calculate step size to ensure all signals are used
                    total_sequences = max(1, int(np.ceil(num_signals / (self.seq_length / 2))))
                    step_size = max(1, int(np.floor((num_signals - self.seq_length) / (total_sequences - 1)))) if total_sequences > 1 else self.seq_length
                    
                    for start_idx in range(0, num_signals - self.seq_length + 1, step_size):
                        end_idx = start_idx + self.seq_length
                        
                        beam_sequences.append({
                            'file_name': file_name,
                            'scan_key': scan_key,
                            'signals': signals[start_idx:end_idx],
                            'annotations': annotations,
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        })
                    
                    # Add the last sequence if needed
                    if (num_signals - self.seq_length) % step_size != 0:
                        beam_sequences.append({
                            'file_name': file_name,
                            'scan_key': scan_key,
                            'signals': signals[-self.seq_length:],
                            'annotations': annotations,
                            'start_idx': num_signals - self.seq_length,
                            'end_idx': num_signals
                        })
        
        return beam_sequences
    
    def visualize_sequence(self, sequence_idx, save_path=None, random_seed=None):
        """
        Visualize a signal sequence with annotations.
        
        Args:
            sequence_idx (int): Index of the sequence to visualize
            save_path (str, optional): Path to save the visualization
            random_seed (int, optional): Random seed for reproducibility
        """
        if sequence_idx >= len(self.all_sequences):
            print(f"Sequence index {sequence_idx} out of range")
            return
            
        sequence_name = list(self.all_sequences.keys())[sequence_idx]
        sequence = self.all_sequences[sequence_name]
        annotations = self.all_annotations[sequence_name]
        
        # Find a scan key that has defects
        scan_keys_with_defects = []
        for scan_key in sequence.keys():
            if scan_key in annotations and len(annotations[scan_key]) > 0:
                scan_keys_with_defects.append(scan_key)
        
        if scan_keys_with_defects:
            # If we have scan keys with defects, choose one randomly
            if random_seed is not None:
                np.random.seed(random_seed)
            scan_key = np.random.choice(scan_keys_with_defects)
        else:
            # If no scan keys have defects, just take the first one
            scan_key = list(sequence.keys())[0]
        
        signals = sequence[scan_key]
        
        # Find signals that contain defects
        signals_with_defects = set()
        if scan_key in annotations:
            for defect in annotations[scan_key]:
                beam_start, beam_end = defect["bbox"][0], defect["bbox"][1]
                for i in range(len(signals)):
                    # Check if this signal index is within the beam range
                    signal_idx = i
                    beam_position = signal_idx / len(signals)
                    if beam_start <= beam_position <= beam_end:
                        signals_with_defects.add(i)
        
        # Choose which signals to display
        max_signals = 10
        if signals_with_defects:
            # Start with signals that have defects
            display_indices = list(signals_with_defects)
            # If we have fewer than max_signals with defects, add some without defects
            if len(display_indices) < max_signals:
                remaining_indices = [i for i in range(len(signals)) if i not in signals_with_defects]
                if remaining_indices:
                    # Randomly select remaining indices
                    additional_indices = np.random.choice(
                        remaining_indices, 
                        size=min(max_signals - len(display_indices), len(remaining_indices)),
                        replace=False
                    )
                    display_indices.extend(additional_indices)
            # If we have more than max_signals with defects, randomly select max_signals
            elif len(display_indices) > max_signals:
                display_indices = np.random.choice(display_indices, size=max_signals, replace=False)
            # Sort the indices for display
            display_indices = sorted(display_indices)
        else:
            # If no signals have defects, just take the first max_signals
            display_indices = list(range(min(max_signals, len(signals))))
        
        plt.figure(figsize=(15, 10))
        
        # Add main title with sequence name and scan key
        plt.suptitle(f"Sequence: {sequence_name}, Scan Key: {scan_key}", fontsize=16)
        
        for i, signal_idx in enumerate(display_indices):
            plt.subplot(len(display_indices), 1, i+1)
            signal = signals[signal_idx]
            plt.plot(signal)
            
            if scan_key in annotations:
                for defect in annotations[scan_key]:
                    # Check if this signal index is within the beam range
                    beam_start, beam_end = defect["bbox"][0], defect["bbox"][1]
                    beam_position = signal_idx / len(signals)
                    
                    if beam_start <= beam_position <= beam_end:
                        # Original values from JSON file
                        original_defect_start = defect["bbox"][2]
                        original_defect_end = defect["bbox"][3]
                        
                        # Calculated values for visualization
                        defect_start = int(original_defect_start * len(signal))
                        defect_end = int(original_defect_end * len(signal))
                        
                        plt.axvspan(defect_start, defect_end, alpha=0.3, color='red')
                        plt.text(defect_start, max(signal), 
                                f"{defect['label']}\nOrig: {original_defect_start:.4f}-{original_defect_end:.4f}\nCalc: {defect_start}-{defect_end}", 
                                fontsize=8)
            
            plt.title(f"Signal {signal_idx}")
            plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class SignalSequenceDataset(Dataset):
    """
    Dataset for signal sequences.
    """
    def __init__(self, sequences_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sequences_file (str): Path to the sequences file
            transform (callable, optional): Optional transform to be applied on a sample
        """
        with open(sequences_file, 'rb') as f:
            import pickle
            self.sequences = pickle.load(f)
            
        self.transform = transform
        
        # Extract all unique labels
        all_labels = set()
        for seq in self.sequences:
            for annot in seq['annotations']:
                all_labels.add(annot['label'])
        
        # Create label mapping
        self.label_map = {label: i for i, label in enumerate(sorted(all_labels))}
        self.label_map['Health'] = len(self.label_map)  # Add 'Health' as the last class
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Extract signals and convert to tensor - signals are already normalized
        signals = torch.tensor(sequence['signals'], dtype=torch.float32)
        
        # Create targets
        targets = []
        for i in range(len(signals)):
            # Default to 'Health' class
            target = {
                'label': self.label_map['Health'],
                'defect_position': torch.zeros(2, dtype=torch.float32)
            }
            
            # Check if there's a defect in this signal
            for annot in sequence['annotations']:
                beam_start, beam_end = annot['bbox'][0], annot['bbox'][1]
                defect_start, defect_end = annot['bbox'][2], annot['bbox'][3]
                
                # Check if this signal index is within the beam range
                if 'start_idx' in sequence:
                    signal_idx = sequence['start_idx'] + i
                    beam_position = signal_idx / len(sequence['signals'])
                    
                    if beam_start <= beam_position <= beam_end:
                        target['label'] = self.label_map[annot['label']]
                        # Store defect start and end positions within the signal
                        target['defect_position'] = torch.tensor([defect_start, defect_end], dtype=torch.float32)
                        break
            
            targets.append(target)
        
        if self.transform:
            signals = self.transform(signals)
            
        return {
            'signals': signals,
            'targets': targets,
            'file_name': sequence['file_name'],
            'scan_key': sequence['scan_key']
        }


if __name__ == "__main__":
    # ds_folder = "WOT-20250522(auto)"
    ds_folder = "/Users/kseni/Documents/GitHub/DefectDetection_viaObjectDetection/D-Fine/ds_manipulations/WOT-20250522(auto)"
    output_folder = "signal_dataset"

    prep = SignalSequencePreparation(ds_folder, output_folder, seq_length=50)
    sequences, annotations = prep.create_signal_sequences()

    # Try different random seeds until we find one that shows defects
    for seed in range(10):
        prep.visualize_sequence(np.random.randint(0, len(prep.all_sequences)), 
                               save_path=os.path.join(output_folder, f"sequence_example_{seed}.png"),
                               random_seed=seed)

    dataset = SignalSequenceDataset(os.path.join(output_folder, "signal_sequences.pkl"))
    print(f"Dataset size: {len(dataset)}")
    print(f"Label map: {dataset.label_map}")

    sample = dataset[0]
    print(f"Sample signals shape: {sample['signals'].shape}")
    print(f"Sample targets: {len(sample['targets'])}")
