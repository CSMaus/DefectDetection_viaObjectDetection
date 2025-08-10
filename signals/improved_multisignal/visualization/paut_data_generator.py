import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Dict

class PAUTDataGenerator:
    """Generate realistic PAUT data for visualization purposes"""
    
    def __init__(self, x_size=50, y_size=30, z_size=320):
        self.x_size = x_size  # Scan positions
        self.y_size = y_size  # Array elements
        self.z_size = z_size  # Time samples (A-scan depth)
        
        # Material properties
        self.material_thickness = 25.0  # mm
        self.sound_velocity = 5900  # m/s for steel
        self.sampling_frequency = 100e6  # 100 MHz
        
        # Defect parameters
        self.defects = []
        
    def add_defect(self, x_pos: int, y_pos: int, depth_start: float, depth_end: float, 
                   amplitude: float = 0.8, defect_type: str = "crack"):
        """Add a defect to the material"""
        self.defects.append({
            'x_pos': x_pos,
            'y_pos': y_pos, 
            'depth_start': depth_start,
            'depth_end': depth_end,
            'amplitude': amplitude,
            'type': defect_type
        })
    
    def generate_clean_ascan(self, depth_mm: float = 25.0) -> np.ndarray:
        """Generate a clean A-scan signal with back wall echo"""
        t = np.linspace(0, 2 * depth_mm / (self.sound_velocity / 1000), self.z_size)
        
        # Back wall echo time
        back_wall_time = 2 * depth_mm / (self.sound_velocity / 1000)
        back_wall_sample = int(back_wall_time * self.sampling_frequency / 1e6)
        
        # Generate signal
        ascan = np.zeros(self.z_size)
        
        # Add back wall echo (Gaussian pulse)
        if back_wall_sample < self.z_size:
            pulse_width = 10
            pulse = 0.6 * np.exp(-((np.arange(self.z_size) - back_wall_sample) / pulse_width) ** 2)
            ascan += pulse
        
        # Add noise
        noise = 0.05 * np.random.randn(self.z_size)
        ascan += noise
        
        return ascan
    
    def generate_defect_ascan(self, defect: Dict, depth_mm: float = 25.0) -> np.ndarray:
        """Generate A-scan with defect echo"""
        ascan = self.generate_clean_ascan(depth_mm)
        
        # Calculate defect echo positions
        defect_start_time = 2 * defect['depth_start'] / (self.sound_velocity / 1000)
        defect_end_time = 2 * defect['depth_end'] / (self.sound_velocity / 1000)
        
        start_sample = int(defect_start_time * self.sampling_frequency / 1e6)
        end_sample = int(defect_end_time * self.sampling_frequency / 1e6)
        
        # Add defect echo
        if start_sample < self.z_size and end_sample < self.z_size:
            defect_length = end_sample - start_sample
            if defect_length > 0:
                # Create defect signature based on type
                if defect['type'] == 'crack':
                    # Sharp, high amplitude echo
                    defect_echo = defect['amplitude'] * np.exp(-((np.arange(defect_length) - defect_length/2) / 3) ** 2)
                elif defect['type'] == 'void':
                    # Broader, complex echo
                    defect_echo = defect['amplitude'] * (
                        0.8 * np.exp(-((np.arange(defect_length) - defect_length/3) / 5) ** 2) +
                        0.4 * np.exp(-((np.arange(defect_length) - 2*defect_length/3) / 4) ** 2)
                    )
                else:
                    # Generic defect
                    defect_echo = defect['amplitude'] * np.exp(-((np.arange(defect_length) - defect_length/2) / 4) ** 2)
                
                ascan[start_sample:start_sample + len(defect_echo)] += defect_echo
        
        return ascan
    
    def generate_3d_data(self) -> np.ndarray:
        """Generate complete 3D PAUT dataset"""
        data_3d = np.zeros((self.x_size, self.y_size, self.z_size))
        
        for x in range(self.x_size):
            for y in range(self.y_size):
                # Check if this position has a defect
                has_defect = False
                for defect in self.defects:
                    # Check if current position is within defect area
                    x_range = range(max(0, defect['x_pos'] - 2), min(self.x_size, defect['x_pos'] + 3))
                    y_range = range(max(0, defect['y_pos'] - 1), min(self.y_size, defect['y_pos'] + 2))
                    
                    if x in x_range and y in y_range:
                        # Generate defect signal with some variation
                        variation = 0.8 + 0.4 * np.random.random()
                        defect_copy = defect.copy()
                        defect_copy['amplitude'] *= variation
                        data_3d[x, y, :] = self.generate_defect_ascan(defect_copy)
                        has_defect = True
                        break
                
                if not has_defect:
                    data_3d[x, y, :] = self.generate_clean_ascan()
        
        return data_3d
    
    def get_signal_sequences(self, data_3d: np.ndarray, sequence_length: int = 5) -> Tuple[List[np.ndarray], List[int]]:
        """Extract signal sequences from 3D data (as done by neural networks)"""
        sequences = []
        labels = []
        
        for x in range(self.x_size - sequence_length + 1):
            for y in range(self.y_size):
                # Extract sequence of signals
                sequence = data_3d[x:x+sequence_length, y, :]
                sequences.append(sequence)
                
                # Check if sequence contains defects
                has_defect = 0
                for defect in self.defects:
                    if (defect['x_pos'] >= x and defect['x_pos'] < x + sequence_length and 
                        abs(defect['y_pos'] - y) <= 1):
                        has_defect = 1
                        break
                
                labels.append(has_defect)
        
        return sequences, labels
    
    def create_sample_dataset(self) -> Dict:
        """Create a sample dataset with various defects"""
        # Add different types of defects
        self.add_defect(x_pos=15, y_pos=10, depth_start=8.0, depth_end=12.0, 
                       amplitude=0.9, defect_type="crack")
        self.add_defect(x_pos=25, y_pos=15, depth_start=15.0, depth_end=18.0, 
                       amplitude=0.7, defect_type="void")
        self.add_defect(x_pos=35, y_pos=20, depth_start=5.0, depth_end=7.0, 
                       amplitude=0.8, defect_type="inclusion")
        
        # Generate 3D data
        data_3d = self.generate_3d_data()
        
        # Extract sequences
        sequences, labels = self.get_signal_sequences(data_3d)
        
        return {
            'data_3d': data_3d,
            'sequences': sequences,
            'labels': labels,
            'defects': self.defects,
            'dimensions': (self.x_size, self.y_size, self.z_size)
        }

def save_sample_data():
    """Generate and save sample PAUT data"""
    generator = PAUTDataGenerator()
    dataset = generator.create_sample_dataset()
    
    # Save as numpy files
    np.save('paut_3d_data.npy', dataset['data_3d'])
    np.save('paut_sequences.npy', np.array(dataset['sequences'], dtype=object))
    np.save('paut_labels.npy', np.array(dataset['labels']))
    
    # Save metadata
    import json
    metadata = {
        'defects': dataset['defects'],
        'dimensions': dataset['dimensions'],
        'num_sequences': len(dataset['sequences']),
        'defect_sequences': sum(dataset['labels'])
    }
    
    with open('paut_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated PAUT dataset:")
    print(f"  3D Data shape: {dataset['data_3d'].shape}")
    print(f"  Number of sequences: {len(dataset['sequences'])}")
    print(f"  Defect sequences: {sum(dataset['labels'])}")
    print(f"  Clean sequences: {len(dataset['labels']) - sum(dataset['labels'])}")

if __name__ == "__main__":
    save_sample_data()
