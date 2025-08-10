import json
import numpy as np
import os
import random
import re
from typing import Dict, List, Tuple, Optional

# DEBUG CONTROL - Set to False to disable all debug prints
DEBUG_PRINTS = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINTS is True"""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

class RealPAUTDataLoader:
    """Load actual PAUT training data from JSON files"""
    
    def __init__(self, json_dir: str):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.loaded_data = {}
        
    def extract_depth_range_from_filename(self, filename: str) -> Tuple[float, float]:
        """Extract depth range from JSON filename (e.g., 'D1.2-12.6' -> (1.2, 12.6))"""
        # Pattern to match depth range like D0-23.92, D1.2-12.6, D-1-10, etc.
        pattern = r'D(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, filename)
        
        if match:
            depth_start = float(match.group(1))
            depth_end = float(match.group(2))
            debug_print(f"Extracted depth range from {filename}: {depth_start} to {depth_end}mm")
            return depth_start, depth_end
        else:
            # Fallback - assume 0 to 100mm if can't parse
            debug_print(f"Warning: Could not parse depth range from {filename}, using default 0-100mm")
            return 0.0, 100.0
    
    def load_json_file(self, filename: str) -> Dict:
        """Load a specific JSON file"""
        if filename in self.loaded_data:
            return self.loaded_data[filename]
            
        file_path = os.path.join(self.json_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.loaded_data[filename] = data
            debug_print(f"Loaded {filename} with {len(data)} beams")
            return data
        except Exception as e:
            debug_print(f"Error loading {filename}: {e}")
            return {}
    
    def calculate_defect_signal_positions(self, defect_start_norm: float, defect_end_norm: float, 
                                        signal_length: int = 320) -> Tuple[int, int]:
        """
        Calculate defect positions in signal samples:
        Defect positions are ALREADY normalized (0.0 to 1.0)
        Just multiply by signal length to get sample indices
        """
        debug_print(f"CALC: Defect normalized positions: {defect_start_norm:.3f} to {defect_end_norm:.3f}")
        
        # Convert normalized positions directly to signal sample indices
        start_sample = int(defect_start_norm * signal_length)
        end_sample = int(defect_end_norm * signal_length)
        
        debug_print(f"CALC: Signal samples: {start_sample} to {end_sample} (signal length: {signal_length})")
        
        # Ensure valid range
        start_sample = max(0, min(signal_length - 1, start_sample))
        end_sample = max(start_sample + 1, min(signal_length, end_sample))
        
        return start_sample, end_sample
    
    def convert_normalized_to_absolute_depth(self, norm_start: float, norm_end: float, 
                                           depth_range: Tuple[float, float]) -> Tuple[float, float]:
        """Convert normalized positions to absolute depth in mm for display"""
        depth_min, depth_max = depth_range
        depth_span = depth_max - depth_min
        
        abs_start = depth_min + (norm_start * depth_span)
        abs_end = depth_min + (norm_end * depth_span)
        
        return abs_start, abs_end
    
    def get_defect_sequences(self, filename: str, seq_length: int = 50) -> List[Dict]:
        """Extract sequences that contain defects from a JSON file"""
        data = self.load_json_file(filename)
        defect_sequences = []
        depth_range = self.extract_depth_range_from_filename(filename)
        
        debug_print(f"Processing {filename} with depth range {depth_range}")
        
        for beam_key, beam_data in data.items():
            debug_print(f"Processing beam {beam_key} with {len(beam_data)} scans")
            
            # Sort scan keys by position
            scan_keys = sorted(beam_data.keys(), key=lambda x: int(x.split('_')[0]))
            
            # Show first few scan keys for debugging
            debug_print(f"First 5 scan keys: {scan_keys[:5]}")
            
            if len(scan_keys) < seq_length:
                continue
                
            # Create sequences
            for i in range(len(scan_keys) - seq_length + 1):
                sequence_keys = scan_keys[i:i + seq_length]
                
                # Check if sequence contains defects
                has_defect = False
                defect_info = []
                signals = []
                labels = []
                
                for scan_key in sequence_keys:
                    scan_data = beam_data[scan_key]
                    
                    # Extract signal data
                    if isinstance(scan_data, list):
                        signal = np.array(scan_data, dtype=np.float32)
                    elif isinstance(scan_data, dict) and 'signal' in scan_data:
                        signal = np.array(scan_data['signal'], dtype=np.float32)
                    else:
                        signal = np.array(scan_data, dtype=np.float32)
                    
                    signals.append(signal)
                    
                    # Extract defect information
                    scan_parts = scan_key.split('_')
                    debug_print(f"Scan key: {scan_key}, parts: {scan_parts}")
                    
                    if len(scan_parts) >= 2 and scan_parts[1] == "Health":
                        labels.append(0)
                        defect_info.append({'has_defect': False, 'start': 0.0, 'end': 0.0, 
                                          'start_sample': 0, 'end_sample': 0})
                        debug_print(f"  -> Clean signal")
                    elif len(scan_parts) >= 3:  # Defect signal: position_DEFECTNAME_range
                        labels.append(1)
                        has_defect = True
                        try:
                            # Extract defect position from scan key
                            defect_range_str = scan_parts[2]  # e.g., "0.2-0.4" (ALREADY NORMALIZED!)
                            defect_name = scan_parts[1]       # e.g., "CRACK", "VOID", etc.
                            
                            debug_print(f"  -> Defect signal: {defect_name}, normalized range: {defect_range_str}")
                            
                            defect_range = defect_range_str.split('-')
                            defect_start_norm = float(defect_range[0])  # Already normalized 0.0-1.0
                            defect_end_norm = float(defect_range[1])    # Already normalized 0.0-1.0
                            
                            debug_print(f"  -> Parsed normalized defect: {defect_start_norm:.3f}-{defect_end_norm:.3f}")
                            
                            # Calculate signal sample positions (no depth conversion needed!)
                            start_sample, end_sample = self.calculate_defect_signal_positions(
                                defect_start_norm, defect_end_norm, len(signal)
                            )
                            
                            # Convert to absolute depth for display purposes only
                            abs_start, abs_end = self.convert_normalized_to_absolute_depth(
                                defect_start_norm, defect_end_norm, depth_range
                            )
                            
                            defect_info.append({
                                'has_defect': True, 
                                'start': abs_start,             # Absolute position (mm) - SHOW IN TEXT
                                'end': abs_end,                 # Absolute position (mm) - SHOW IN TEXT
                                'start_norm': defect_start_norm, # Normalized position - FOR REFERENCE
                                'end_norm': defect_end_norm,     # Normalized position - FOR REFERENCE
                                'start_sample': start_sample,   # Signal sample index - USE FOR BBOX
                                'end_sample': end_sample,       # Signal sample index - USE FOR BBOX
                                'scan_key': scan_key,
                                'defect_name': defect_name
                            })
                            debug_print(f"  -> Added defect info: {abs_start:.1f}-{abs_end:.1f}mm (norm: {defect_start_norm:.3f}-{defect_end_norm:.3f}) -> samples {start_sample}-{end_sample}")
                            
                        except Exception as e:
                            debug_print(f"Error parsing defect from {scan_key}: {e}")
                            defect_info.append({'has_defect': True, 'start': 0.0, 'end': 0.0,
                                              'start_sample': 0, 'end_sample': 0})
                    else:
                        # Unknown format
                        labels.append(0)
                        defect_info.append({'has_defect': False, 'start': 0.0, 'end': 0.0, 
                                          'start_sample': 0, 'end_sample': 0})
                        debug_print(f"  -> Unknown format, treating as clean")
                
                # Only include sequences with defects
                if has_defect:
                    defect_sequences.append({
                        'filename': filename,
                        'beam_key': beam_key,
                        'signals': np.array(signals),
                        'labels': np.array(labels),
                        'defect_info': defect_info,
                        'sequence_keys': sequence_keys,
                        'depth_range': depth_range
                    })
                    debug_print(f"Added sequence with {sum(labels)} defects out of {len(labels)} signals")
                    break  # Just take first sequence with defects for now
            
            if defect_sequences:  # Found at least one sequence
                break
        
        debug_print(f"Total defect sequences found: {len(defect_sequences)}")
        return defect_sequences
    
    def get_sample_defect_sequence(self, seq_length: int = 50) -> Optional[Dict]:
        """Get one sample defect sequence for visualization"""
        # Try different files to find one with good defect examples
        for filename in self.json_files[:5]:
            debug_print(f"Trying file: {filename}")
            sequences = self.get_defect_sequences(filename, seq_length)
            if sequences:
                return sequences[0]  # Return first sequence found
        return None
    
    def get_mixed_sequence_for_attention(self, seq_length: int = 50) -> Optional[Dict]:
        """Get a sequence with mixed defect/clean signals for attention visualization"""
        return self.get_sample_defect_sequence(seq_length)  # Simplified for now
    
    def get_signal_comparison(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Get clean vs defect signal comparison from real data"""
        sequence = self.get_sample_defect_sequence(seq_length=50)
        if not sequence:
            debug_print("No defect sequences found!")
            return None, None, {}
        
        debug_print(f"Found sequence with {len(sequence['signals'])} signals")
        
        # Find clean and defect signals
        clean_signal = None
        defect_signal = None
        defect_info = None
        
        for i, (signal, label, info) in enumerate(zip(sequence['signals'], 
                                                     sequence['labels'], 
                                                     sequence['defect_info'])):
            debug_print(f"Signal {i}: label={label}, has_defect={info.get('has_defect', False)}, start={info.get('start', 0)}")
            
            if label == 0 and clean_signal is None:
                clean_signal = signal
                debug_print(f"  -> Selected as clean signal")
            elif label == 1 and defect_signal is None and info.get('start', 0) > 0:
                defect_signal = signal
                defect_info = info
                defect_info['depth_range'] = sequence['depth_range']  # Add depth range info
                debug_print(f"  -> Selected as defect signal: {info['start']:.1f}-{info['end']:.1f}mm (norm: {info.get('start_norm', 0):.3f}-{info.get('end_norm', 0):.3f}), samples {info['start_sample']}-{info['end_sample']}")
                break
        
        if defect_signal is None:
            debug_print("No valid defect signal found!")
        if clean_signal is None:
            debug_print("No clean signal found!")
            
        return clean_signal, defect_signal, defect_info

def get_real_data_sample():
    """Convenience function to get real data sample"""
    json_dir = "/Users/kseni/Documents/GitHub/DefectDetection_viaObjectDetection/signals/improved_multisignal/json_data"
    loader = RealPAUTDataLoader(json_dir)
    
    # Try different files to get variety
    sample_files = [
        "WOT D01-D03_01_Ch-0_D1.2-12.6.json",
        "WOT D33-D36_03_Ch-0_D0-15.5.json", 
        "WOT D64-D67_01_Ch-0_D2-12.json",
        "WOT D22-D24_03_Ch-0_D70-76.json"
    ]
    
    for filename in sample_files:
        if filename in loader.json_files:
            debug_print(f"Using file: {filename}")
            return loader, filename
    
    # Fallback to first available file
    if loader.json_files:
        debug_print(f"Using fallback file: {loader.json_files[0]}")
        return loader, loader.json_files[0]
    
    return None, None

if __name__ == "__main__":
    # Test the data loader
    loader, filename = get_real_data_sample()
    if loader and filename:
        debug_print(f"Testing with file: {filename}")
        
        # Test signal comparison
        clean_signal, defect_signal, defect_info = loader.get_signal_comparison()
        if defect_info:
            depth_range = defect_info['depth_range']
            debug_print(f"FINAL RESULT:")
            debug_print(f"  Defect position: {defect_info['start']:.1f}-{defect_info['end']:.1f}mm")
            debug_print(f"  Normalized: {defect_info.get('start_norm', 0):.3f}-{defect_info.get('end_norm', 0):.3f}")
            debug_print(f"  Depth range: {depth_range[0]:.1f}-{depth_range[1]:.1f}mm")
            debug_print(f"  Signal samples: {defect_info['start_sample']}-{defect_info['end_sample']}")
        else:
            debug_print("NO DEFECT INFO FOUND!")
    else:
        debug_print("No data files found!")
