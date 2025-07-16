#!/usr/bin/env python3
"""
Test script to verify that the flexible dataset loader correctly handles both old and new JSON file formats.
"""

import os
import sys
from defect_focused_dataset_flexible import FlexibleDefectFocusedJsonSignalDataset

def test_file_format_detection():
    """Test the file format detection function"""
    print("Testing file format detection...")
    
    # Create a dummy dataset instance to access the method
    dummy_dataset = FlexibleDefectFocusedJsonSignalDataset.__new__(FlexibleDefectFocusedJsonSignalDataset)
    
    # Test cases
    test_cases = [
        ("WOT_D33-D36_01_Ch-0_D0-14.json", "old"),
        ("WOT_D33-D36_01_Ch-0-S350_400-D0_14.json", "new"),
        ("beam_data_Ch-1_D5-20.json", "old"),
        ("beam_data_Ch-1-S100_200-D5_20.json", "new"),
        ("test_file.json", "old"),  # Default to old if no S pattern
    ]
    
    for filename, expected_format in test_cases:
        detected_format = dummy_dataset._detect_file_format(filename)
        status = "✓" if detected_format == expected_format else "✗"
        print(f"  {status} {filename} -> {detected_format} (expected: {expected_format})")
    
    print()

def test_defect_extraction_from_filename():
    """Test defect extraction from filename"""
    print("Testing defect extraction from filename...")
    
    # Create a dummy dataset instance to access the method
    dummy_dataset = FlexibleDefectFocusedJsonSignalDataset.__new__(FlexibleDefectFocusedJsonSignalDataset)
    
    # Test cases
    test_cases = [
        ("WOT_D33-D36_01_Ch-0_D0-14.json", True, 0.0, 14.0),
        ("WOT_D33-D36_01_Ch-0-S350_400-D0_14.json", True, 0.0, 14.0),
        ("beam_data_Ch-1_D5-20.json", True, 5.0, 20.0),
        ("beam_data_Ch-1-S100_200-D5_20.json", True, 5.0, 20.0),
        ("healthy_data.json", False, None, None),
    ]
    
    for filename, expected_has_defect, expected_start, expected_end in test_cases:
        has_defect, start, end = dummy_dataset._extract_defect_from_filename(filename)
        
        if expected_has_defect:
            status = "✓" if (has_defect and start == expected_start and end == expected_end) else "✗"
            print(f"  {status} {filename} -> defect: {has_defect}, range: [{start}, {end}] (expected: [{expected_start}, {expected_end}])")
        else:
            status = "✓" if not has_defect else "✗"
            print(f"  {status} {filename} -> defect: {has_defect} (expected: {expected_has_defect})")
    
    print()

def test_scan_key_parsing():
    """Test scan key parsing for defect information"""
    print("Testing scan key parsing...")
    
    # Create a dummy dataset instance to access the method
    dummy_dataset = FlexibleDefectFocusedJsonSignalDataset.__new__(FlexibleDefectFocusedJsonSignalDataset)
    
    # Test cases
    test_cases = [
        ("0_Health", False, None, None),
        ("1_Defect_0-14", True, 0.0, 14.0),
        ("2_Defect_5_20", True, 5.0, 20.0),  # New format with underscore
        ("3_Defect_10", True, 10.0, 11.0),   # Single number case
        ("4_Unknown", False, None, None),     # Unknown format
    ]
    
    for scan_key, expected_is_defect, expected_start, expected_end in test_cases:
        is_defect, start, end = dummy_dataset._extract_defect_info_from_scan_key(scan_key)
        
        if expected_is_defect:
            status = "✓" if (is_defect and start == expected_start and end == expected_end) else "✗"
            print(f"  {status} {scan_key} -> defect: {is_defect}, range: [{start}, {end}] (expected: [{expected_start}, {expected_end}])")
        else:
            status = "✓" if not is_defect else "✗"
            print(f"  {status} {scan_key} -> defect: {is_defect} (expected: {expected_is_defect})")
    
    print()

def test_dataset_loading():
    """Test actual dataset loading if JSON files are available"""
    print("Testing dataset loading...")
    
    json_dir = "json_data/"
    
    if not os.path.exists(json_dir):
        print(f"  ⚠ JSON directory '{json_dir}' not found. Skipping dataset loading test.")
        return
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"  ⚠ No JSON files found in '{json_dir}'. Skipping dataset loading test.")
        return
    
    print(f"  Found {len(json_files)} JSON files in '{json_dir}'")
    
    try:
        # Try to create the dataset
        dataset = FlexibleDefectFocusedJsonSignalDataset(
            json_dir=json_dir,
            seq_length=10,  # Small sequence length for testing
            min_defects_per_sequence=1
        )
        
        print(f"  ✓ Successfully loaded dataset with {len(dataset)} sequences")
        
        if len(dataset) > 0:
            # Test getting a sample
            signals, labels, defect_positions = dataset[0]
            print(f"  ✓ Sample shape: signals={signals.shape}, labels={labels.shape}, defect_positions={defect_positions.shape}")
            
            # Check for defects in the sample
            defect_count = labels.sum().item()
            print(f"  ✓ Sample contains {defect_count} defects")
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
    
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("FLEXIBLE DATASET LOADER TESTS")
    print("=" * 60)
    print()
    
    test_file_format_detection()
    test_defect_extraction_from_filename()
    test_scan_key_parsing()
    test_dataset_loading()
    
    print("=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
