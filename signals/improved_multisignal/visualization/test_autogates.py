"""
Test script to verify autogates function and preview data
"""

import numpy as np
import matplotlib.pyplot as plt
from autogates_func import synth_dscan, row_statistic, gradients_1st_2nd, find_peaks_by_second_derivative

def test_autogates():
    # Generate synthetic D-scan data
    print("Generating synthetic D-scan data...")
    dscan_data = synth_dscan(height=320, width=301, seed=42)
    print(f"D-scan shape: {dscan_data.shape}")
    print(f"D-scan range: [{dscan_data.min():.3f}, {dscan_data.max():.3f}]")
    
    # Calculate mean signal
    print("\nCalculating mean signal...")
    mean_signal = row_statistic(dscan_data, mode="mean")
    print(f"Mean signal shape: {mean_signal.shape}")
    print(f"Mean signal range: [{mean_signal.min():.3f}, {mean_signal.max():.3f}]")
    
    # Calculate derivatives
    print("\nCalculating derivatives...")
    d1, d2 = gradients_1st_2nd(mean_signal)
    print(f"1st derivative range: [{d1.min():.3f}, {d1.max():.3f}]")
    print(f"2nd derivative range: [{d2.min():.3f}, {d2.max():.3f}]")
    
    # Find peaks
    print("\nFinding peaks...")
    peaks, _, _ = find_peaks_by_second_derivative(mean_signal)
    print(f"Found {len(peaks)} peaks:")
    for i, (start, end) in enumerate(peaks):
        print(f"  Peak {i+1}: indices {start} to {end}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # D-scan heatmap
    im = axes[0, 0].imshow(dscan_data, aspect='auto', cmap='jet', origin='upper')
    axes[0, 0].set_title('D-scan Heatmap')
    axes[0, 0].set_xlabel('Width (samples)')
    axes[0, 0].set_ylabel('Height (samples)')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Mean signal
    axes[0, 1].plot(mean_signal, 'y-', linewidth=2, label='Mean Signal')
    axes[0, 1].set_title('Mean Signal')
    axes[0, 1].set_xlabel('Row Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # First derivative
    axes[1, 0].plot(d1, 'g-', linewidth=2, label='1st Derivative')
    axes[1, 0].set_title('First Derivative')
    axes[1, 0].set_xlabel('Row Index')
    axes[1, 0].set_ylabel('Gradient')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Second derivative with threshold and peaks
    axes[1, 1].plot(d2, 'b-', linewidth=2, label='2nd Derivative')
    
    # Add threshold line
    threshold = d2.max() / 4.0
    axes[1, 1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                       label=f'Threshold = {threshold:.3f}')
    
    # Highlight peaks
    for i, (start, end) in enumerate(peaks):
        axes[1, 1].axvspan(start, end, alpha=0.3, color='yellow', 
                          label='Gate' if i == 0 else "")
    
    axes[1, 1].set_title('Second Derivative with Peak Detection')
    axes[1, 1].set_xlabel('Row Index')
    axes[1, 1].set_ylabel('Second Gradient')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('autogates_test_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTest completed successfully!")
    print(f"Preview saved as 'autogates_test_preview.png'")

if __name__ == "__main__":
    test_autogates()
