import torch
import numpy as np
import matplotlib.pyplot as plt
from realistic_noise_augmentation import RealisticNoiseAugmentation

def test_noise_augmentation():
    """Test the realistic noise augmentation on sample signals"""
    
    # Create sample signal sequence (similar to your data)
    batch_size = 2
    num_signals = 30
    signal_length = 320
    
    # Generate clean test signals
    t = np.linspace(0, 1, signal_length)
    clean_signals = torch.zeros(batch_size, num_signals, signal_length)
    
    for b in range(batch_size):
        for s in range(num_signals):
            # Create different signal patterns
            freq1 = 5 + s * 0.5
            freq2 = 12 + s * 0.3
            amplitude = 1.0 + 0.1 * np.sin(s)
            
            signal = amplitude * (np.sin(2 * np.pi * freq1 * t) + 
                                0.3 * np.sin(2 * np.pi * freq2 * t) +
                                0.1 * np.sin(2 * np.pi * 25 * t))
            
            clean_signals[b, s] = torch.tensor(signal, dtype=torch.float32)
    
    # Initialize noise augmenter
    noise_augmenter = RealisticNoiseAugmentation(augment_probability=1.0)  # Always augment for testing
    
    # Apply noise augmentation
    noisy_signals = noise_augmenter.augment_sequence(clean_signals)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Show first batch, first few signals
    signals_to_show = min(6, num_signals)
    
    for i in range(signals_to_show):
        plt.subplot(2, 3, i + 1)
        
        # Plot clean and noisy signals
        plt.plot(clean_signals[0, i].numpy(), label='Clean', alpha=0.7, linewidth=1)
        plt.plot(noisy_signals[0, i].numpy(), label='With Realistic Noise', alpha=0.8, linewidth=1)
        
        plt.title(f'Signal {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate SNR
        clean = clean_signals[0, i].numpy()
        noisy = noisy_signals[0, i].numpy()
        noise = noisy - clean
        
        signal_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            plt.text(0.02, 0.98, f'SNR: {snr_db:.1f} dB', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Realistic Noise Augmentation Test', y=1.02, fontsize=14)
    plt.show()
    
    # Print statistics
    print("Noise Augmentation Test Results:")
    print(f"Original signal shape: {clean_signals.shape}")
    print(f"Augmented signal shape: {noisy_signals.shape}")
    
    # Check how many signals were actually augmented
    differences = torch.abs(noisy_signals - clean_signals).sum(dim=2)  # Sum over signal length
    augmented_mask = differences > 1e-6  # Threshold for detecting changes
    
    total_signals = batch_size * num_signals
    augmented_count = augmented_mask.sum().item()
    
    print(f"Total signals: {total_signals}")
    print(f"Augmented signals: {augmented_count}")
    print(f"Augmentation rate: {augmented_count/total_signals*100:.1f}%")
    
    # Analyze noise characteristics
    for b in range(batch_size):
        batch_augmented = augmented_mask[b].sum().item()
        print(f"Batch {b}: {batch_augmented}/{num_signals} signals augmented")


def test_individual_noise_types():
    """Test individual noise types to understand their characteristics"""
    
    # Create a clean test signal
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    
    augmenter = RealisticNoiseAugmentation()
    
    plt.figure(figsize=(15, 12))
    
    # Test different noise types with consistent amplitudes
    noise_tests = [
        ('Original Clean Signal', clean_signal, 'blue'),
        ('Pink Noise (1/f) - Balanced', clean_signal + augmenter.generate_colored_noise(1000, 'pink', 0.06), 'red'),
        ('Brown Noise (1/f²) - Balanced', clean_signal + augmenter.generate_colored_noise(1000, 'brown', 0.06), 'brown'),
        ('White Noise - Balanced', clean_signal + augmenter.generate_colored_noise(1000, 'white', 0.06), 'gray'),
        ('Burst/Impulse Noise', clean_signal + augmenter.generate_burst_noise(1000, 0.02, 0.3), 'orange'),
        ('Baseline Drift', clean_signal + augmenter.generate_baseline_drift(1000, 0.15), 'green'),
        ('Harmonic Interference', clean_signal + augmenter.generate_harmonic_interference(1000), 'purple'),
        ('Combined Realistic', augmenter.apply_realistic_noise(clean_signal), 'black')
    ]
    
    for i, (title, signal, color) in enumerate(noise_tests):
        plt.subplot(2, 4, i + 1)
        plt.plot(signal, color=color, linewidth=1)
        plt.title(title, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if i > 0:  # Calculate SNR for noisy signals
            noise = signal - clean_signal
            signal_power = np.mean(clean_signal**2)
            noise_power = np.mean(noise**2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                plt.text(0.02, 0.98, f'SNR: {snr_db:.1f} dB', 
                        transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Individual Realistic Noise Types', y=1.02, fontsize=14)
    plt.show()


if __name__ == "__main__":
    print("Testing realistic noise augmentation...")
    test_noise_augmentation()
    
    print("\nTesting individual noise types...")
    test_individual_noise_types()
    
    print("\nNoise augmentation tests completed!")
