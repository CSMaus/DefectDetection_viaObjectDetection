import torch
import numpy as np
import random
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


class RealisticNoiseAugmentation:
    """
    Realistic noise augmentation for signal data that mimics real-world noise patterns:
    - Colored noise (1/f, pink, brown noise)
    - Burst noise (impulse noise)
    - Baseline drift
    - Harmonic interference
    - Quantization noise
    - Thermal noise with frequency characteristics
    """
    
    def __init__(self, augment_probability=0.3):
        self.augment_probability = augment_probability
    
    def generate_colored_noise(self, length, noise_type='pink', amplitude=0.1):
        """Generate colored noise (pink, brown, blue) instead of white noise"""
        # Generate white noise
        white_noise = np.random.randn(length)
        
        if noise_type == 'pink':
            # Pink noise (1/f noise) - common in electronic systems
            # Apply 1/f filter in frequency domain
            freqs = np.fft.fftfreq(length)
            freqs[0] = 1e-10  # Avoid division by zero
            filter_response = 1.0 / np.sqrt(np.abs(freqs))
            filter_response[0] = 0
            
            white_fft = np.fft.fft(white_noise)
            pink_fft = white_fft * filter_response
            pink_noise = np.real(np.fft.ifft(pink_fft))
            
        elif noise_type == 'brown':
            # Brown noise (1/f^2 noise) - Brownian motion
            freqs = np.fft.fftfreq(length)
            freqs[0] = 1e-10
            filter_response = 1.0 / np.abs(freqs)
            filter_response[0] = 0
            
            white_fft = np.fft.fft(white_noise)
            brown_fft = white_fft * filter_response
            brown_noise = np.real(np.fft.ifft(brown_fft))
            pink_noise = brown_noise
            
        elif noise_type == 'blue':
            # Blue noise (f noise) - high frequency emphasis
            freqs = np.fft.fftfreq(length)
            filter_response = np.sqrt(np.abs(freqs))
            
            white_fft = np.fft.fft(white_noise)
            blue_fft = white_fft * filter_response
            blue_noise = np.real(np.fft.ifft(blue_fft))
            pink_noise = blue_noise
            
        else:  # white noise
            pink_noise = white_noise
        
        # Normalize and scale with amplitude correction for different noise types
        if noise_type == 'brown':
            # Brown noise tends to have lower amplitude, boost it
            pink_noise = pink_noise / np.std(pink_noise) * amplitude * 1.3
        elif noise_type == 'pink':
            # Pink noise amplitude is good as reference
            pink_noise = pink_noise / np.std(pink_noise) * amplitude
        else:  # white or blue
            # White/blue noise can be more intense, reduce slightly
            pink_noise = pink_noise / np.std(pink_noise) * amplitude * 0.8
            
        return pink_noise
    
    def generate_burst_noise(self, length, burst_probability=0.02, burst_amplitude=0.3):
        """Generate impulse/burst noise - sudden spikes common in real systems"""
        noise = np.zeros(length)
        
        # Random burst locations
        burst_mask = np.random.random(length) < burst_probability
        
        # Generate bursts with varying amplitudes and widths
        burst_indices = np.where(burst_mask)[0]
        
        for idx in burst_indices:
            # Burst width (1-5 samples)
            width = random.randint(1, 5)
            amplitude = random.uniform(-burst_amplitude, burst_amplitude)
            
            # Apply burst
            start_idx = max(0, idx - width//2)
            end_idx = min(length, idx + width//2 + 1)
            noise[start_idx:end_idx] += amplitude
        
        return noise
    
    def generate_baseline_drift(self, length, drift_amplitude=0.05):
        """Generate slow baseline drift - common in sensors and electronics"""
        # Generate low-frequency sinusoidal drift
        t = np.linspace(0, 1, length)
        
        # Multiple frequency components for realistic drift
        drift = 0
        for freq in [0.5, 1.2, 2.1]:  # Low frequencies
            phase = random.uniform(0, 2*np.pi)
            amplitude = random.uniform(0.3, 1.0) * drift_amplitude
            drift += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add very slow trend
        trend_coeff = random.uniform(-drift_amplitude/2, drift_amplitude/2)
        drift += trend_coeff * t
        
        return drift
    
    def generate_harmonic_interference(self, length, sampling_rate=1000, interference_freqs=[50, 60, 120]):
        """Generate harmonic interference (power line, switching frequencies)"""
        t = np.linspace(0, length/sampling_rate, length)
        interference = np.zeros(length)
        
        for freq in interference_freqs:
            if random.random() < 0.4:  # 40% chance for each frequency
                amplitude = random.uniform(0.01, 0.05)
                phase = random.uniform(0, 2*np.pi)
                interference += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        return interference
    
    def generate_quantization_noise(self, signal, bits=12):
        """Generate quantization noise from ADC"""
        # Simulate ADC quantization
        signal_range = np.max(signal) - np.min(signal)
        if signal_range == 0:
            return np.zeros_like(signal)
        
        quantization_step = signal_range / (2**bits)
        quantized = np.round(signal / quantization_step) * quantization_step
        quantization_noise = signal - quantized
        
        return quantization_noise
    
    def add_thermal_noise(self, signal, snr_db=40):
        """Add thermal noise with realistic SNR"""
        signal_power = np.mean(signal**2)
        if signal_power == 0:
            return np.zeros_like(signal)
        
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        thermal_noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
        
        return thermal_noise
    
    def apply_realistic_noise(self, signal):
        """Apply combination of realistic noise types to a single signal"""
        if isinstance(signal, torch.Tensor):
            signal_np = signal.cpu().numpy()
            was_tensor = True
        else:
            signal_np = signal
            was_tensor = False
        
        length = len(signal_np)
        noisy_signal = signal_np.copy()
        
        # Randomly select which noise types to apply
        noise_types = []
        
        # Colored noise (most common) - Use primarily pink noise with consistent amplitude
        if random.random() < 0.8:
            # Use pink noise as primary choice (most realistic for signal processing)
            if random.random() < 0.7:
                noise_type = 'pink'
                amplitude = random.uniform(0.04, 0.07)  # Consistent amplitude range for pink
            elif random.random() < 0.8:
                noise_type = 'brown' 
                amplitude = random.uniform(0.06, 0.09)  # Higher amplitude for brown to match pink
            else:
                noise_type = 'white'
                amplitude = random.uniform(0.03, 0.06)  # Lower amplitude for white
            
            colored_noise = self.generate_colored_noise(length, noise_type, amplitude)
            noisy_signal += colored_noise
            noise_types.append(f'{noise_type}_noise')
        
        # Burst noise (less common but impactful)
        if random.random() < 0.3:
            burst_noise = self.generate_burst_noise(length, 
                                                  burst_probability=random.uniform(0.005, 0.02),
                                                  burst_amplitude=random.uniform(0.1, 0.3))
            noisy_signal += burst_noise
            noise_types.append('burst_noise')
        
        # Baseline drift (very common in real systems)
        if random.random() < 0.6:
            drift = self.generate_baseline_drift(length, 
                                               drift_amplitude=random.uniform(0.02, 0.1))
            noisy_signal += drift
            noise_types.append('baseline_drift')
        
        # Harmonic interference (power line, switching)
        if random.random() < 0.4:
            interference = self.generate_harmonic_interference(length)
            noisy_signal += interference
            noise_types.append('harmonic_interference')
        
        # Quantization noise (always present in digital systems)
        if random.random() < 0.5:
            quant_noise = self.generate_quantization_noise(noisy_signal, 
                                                         bits=random.randint(10, 14))
            noisy_signal += quant_noise
            noise_types.append('quantization_noise')
        
        # Thermal noise (always present)
        if random.random() < 0.7:
            thermal_noise = self.add_thermal_noise(signal_np, 
                                                 snr_db=random.uniform(35, 50))
            noisy_signal += thermal_noise
            noise_types.append('thermal_noise')
        
        if was_tensor:
            return torch.tensor(noisy_signal, dtype=signal.dtype, device=signal.device)
        else:
            return noisy_signal
    
    def augment_sequence(self, sequence):
        """
        Augment a sequence of signals with realistic noise
        
        Args:
            sequence: torch.Tensor of shape [num_signals, signal_length] or [batch, num_signals, signal_length]
        
        Returns:
            Augmented sequence with same shape
        """
        if random.random() > self.augment_probability:
            return sequence  # No augmentation
        
        if len(sequence.shape) == 3:  # Batch dimension
            batch_size, num_signals, signal_length = sequence.shape
            augmented = sequence.clone()
            
            for b in range(batch_size):
                # Randomly select which signals in the sequence to augment
                num_to_augment = random.randint(1, max(1, num_signals // 3))  # Augment 1 to 1/3 of signals
                signals_to_augment = random.sample(range(num_signals), num_to_augment)
                
                for s in signals_to_augment:
                    augmented[b, s] = self.apply_realistic_noise(sequence[b, s])
            
            return augmented
            
        elif len(sequence.shape) == 2:  # No batch dimension
            num_signals, signal_length = sequence.shape
            augmented = sequence.clone()
            
            # Randomly select which signals to augment
            num_to_augment = random.randint(1, max(1, num_signals // 3))
            signals_to_augment = random.sample(range(num_signals), num_to_augment)
            
            for s in signals_to_augment:
                augmented[s] = self.apply_realistic_noise(sequence[s])
            
            return augmented
        
        else:
            raise ValueError(f"Unexpected sequence shape: {sequence.shape}")


# Convenience function for easy integration
def augment_with_realistic_noise(sequence, augment_probability=0.3):
    """
    Convenience function to apply realistic noise augmentation
    
    Args:
        sequence: Signal sequence to augment
        augment_probability: Probability of applying augmentation
    
    Returns:
        Augmented sequence
    """
    augmenter = RealisticNoiseAugmentation(augment_probability)
    return augmenter.augment_sequence(sequence)


# Test function to visualize different noise types
def test_noise_types():
    """Test function to visualize different realistic noise types"""
    import matplotlib.pyplot as plt
    
    # Generate clean test signal
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    
    augmenter = RealisticNoiseAugmentation()
    
    plt.figure(figsize=(15, 12))
    
    # Clean signal
    plt.subplot(3, 3, 1)
    plt.plot(clean_signal)
    plt.title('Clean Signal')
    plt.grid(True)
    
    # Different noise types
    noise_examples = [
        ('Pink Noise', augmenter.generate_colored_noise(1000, 'pink', 0.1)),
        ('Brown Noise', augmenter.generate_colored_noise(1000, 'brown', 0.1)),
        ('Burst Noise', augmenter.generate_burst_noise(1000, 0.02, 0.3)),
        ('Baseline Drift', augmenter.generate_baseline_drift(1000, 0.1)),
        ('Harmonic Interference', augmenter.generate_harmonic_interference(1000)),
        ('Quantization Noise', augmenter.generate_quantization_noise(clean_signal, 8)),
        ('Thermal Noise', augmenter.add_thermal_noise(clean_signal, 30)),
        ('Combined Realistic Noise', augmenter.apply_realistic_noise(clean_signal) - clean_signal)
    ]
    
    for i, (title, noise) in enumerate(noise_examples, 2):
        plt.subplot(3, 3, i)
        if 'Noise' in title and title != 'Combined Realistic Noise':
            plt.plot(noise)
        else:
            plt.plot(clean_signal + noise)
        plt.title(title)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_noise_types()
