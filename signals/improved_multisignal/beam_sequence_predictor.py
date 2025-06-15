import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import math


from improved_model import ImprovedMultiSignalClassifier

JSON_FILE_PATH = 'json_data/WOT-D456_A4_003_Ch-0_D0.5-10.json'
BEAM_INDEX = 29
MODEL_PATH = 'models/improved_model_20250615_033905/best_model.pth'
SEQ_LENGTH = 50

def load_model(model_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    signal_length = 320
    hidden_sizes = [128, 64, 32]
    num_heads = 8
    num_transformer_layers = 4
    
    model = ImprovedMultiSignalClassifier(
        signal_length=signal_length,
        hidden_sizes=hidden_sizes,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model, device

def load_sequences(json_path=JSON_FILE_PATH, seq_len=SEQ_LENGTH):
    """Load sequences from a JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    sequences_by_beams = {}
    labels_by_beams = {}
    defects_by_beams = {}
    for beam_key in data.keys():
        beam_idx = int(beam_key.split('_')[1])
        sequences_by_beams[str(beam_idx)] = {}
        labels_by_beams[str(beam_idx)] = {}
        defects_by_beams[str(beam_idx)] = {}

        scans_keys_sorted = sorted(data[beam_key].keys(), key=lambda x: int(x.split('_')[0]))
        # num_of_seqs_for_beam = math.ceil(len(scans_keys_sorted) / seq_len)
        all_scans_for_beam = {}
        all_lbls_for_beam = {}
        all_defects_for_beam = {}
        for scan_key in scans_keys_sorted:
            scan_data = data[beam_key][scan_key]
            scan_idx = int(scan_key.split('_')[0])

            all_scans_for_beam[str(scan_idx)] = scan_data
            if scan_key.split('_')[1] == "Health":
                # labels_by_beams[str(beam_idx)][str(scan_idx)] = 0
                # defects_by_beams[str(beam_idx)][str(scan_idx)] = [None, None]
                all_lbls_for_beam[str(scan_idx)] = 0
                all_defects_for_beam[str(scan_idx)] = [None, None]
            else:
                # labels_by_beams[str(beam_idx)][str(scan_idx)] = 1
                all_lbls_for_beam[str(scan_idx)] = 1
                defect_range = scan_key.split('_')[2].split('-')  # .split('.')[0]
                defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                # defects_by_beams[str(beam_idx)][str(scan_idx)] = [defect_start, defect_end]
                all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]

        # now from all_scans_for_beam we need to create num_of_seqs_for_beam number of sequences
        # first sequences will go just with increasing indexes, but for the last if the number of remaining
        # scans is less than seq_len, then we have to start from the scan_key from which to the last scan_key in scans_keys_sorted
        # we will be able to receive sequence with len == seq_len
        num_of_seqs_for_beam = math.ceil(len(scans_keys_sorted) / seq_len)
        for i in range(num_of_seqs_for_beam):
            # where the "i" is the index of the sequence for current beam_idx
            sequence = []
            labels = []
            defects = []

            if i < num_of_seqs_for_beam - 1:
                start_idx = i * seq_len
                end_idx = start_idx + seq_len
            else:
                start_idx = len(scans_keys_sorted) - seq_len - 1
                end_idx = len(scans_keys_sorted)

            for j in range(start_idx, end_idx):
                scan_data = all_scans_for_beam[str(j)]
                labels.append(all_lbls_for_beam[str(j)])
                defects.append(all_defects_for_beam[str(j)])
                sequence.append(scan_data)

            sequences_by_beams[str(beam_idx)][str(i)] = sequence
            labels_by_beams[str(beam_idx)][str(i)] = labels
            defects_by_beams[str(beam_idx)][str(i)] = defects


    return sequences_by_beams, labels_by_beams, defects_by_beams

def extract_beam_signals(json_file_path, beam_index):
    """Extract signals for a specific beam from the JSON file"""
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get beam keys and select the specified beam
    beam_keys = list(data.keys())
    if beam_index >= len(beam_keys):
        raise ValueError(f"Beam index {beam_index} out of range. Available beams: 0-{len(beam_keys)-1}")
    
    beam_key = beam_keys[beam_index]
    beam_data = data[beam_key]
    
    print(f"Processing beam: {beam_key}")
    
    all_signals = []
    all_labels = []
    all_defect_positions = []
    all_signal_keys = []
    
    signal_files = []
    for scan_key in beam_data.keys():
        scan_data = beam_data[scan_key]
        if isinstance(scan_data, dict):
            for signal_key in scan_data.keys():
                signal_files.append((scan_key, signal_key))
        elif isinstance(scan_data, list):
            for i, _ in enumerate(scan_data):
                signal_files.append((scan_key, str(i)))
    
    try:
        signal_files.sort(key=lambda x: int(round(float(x[0].split('_')[0]))) 
                        if '_' in x[0] and x[0].split('_')[0].replace('.', '', 1).isdigit() else 0)
        print(f"Signals sorted by scan index")
    except Exception as e:
        print(f"Warning: Could not sort by scan index: {e}")
        print("Using original order")
    
    # Process each signal in sorted order
    for scan_key, signal_key in signal_files:
        try:
            # Get signal data
            if isinstance(beam_data[scan_key], dict):
                signal_data = beam_data[scan_key].get(signal_key)
            elif isinstance(beam_data[scan_key], list):
                idx = int(signal_key)
                if idx < len(beam_data[scan_key]):
                    signal_data = beam_data[scan_key][idx]
                else:
                    continue
            else:
                continue
            
            if isinstance(signal_data, list):
                signal = np.array(signal_data, dtype=np.float32)
            elif isinstance(signal_data, dict) and 'signal' in signal_data:
                signal = np.array(signal_data['signal'], dtype=np.float32)
            else:
                signal = np.array(signal_data, dtype=np.float32)
            
            if signal is None or signal.size == 0:
                continue
            
            all_signals.append(signal)
            all_signal_keys.append(f"{scan_key}_{signal_key}")
            
            if '_' in signal_key:
                defect_name = signal_key.split('_')[1]
                if defect_name == 'Health':
                    all_labels.append(0.0)
                    all_defect_positions.append([0.0, 0.0])
                else:
                    try:
                        defect_range = signal_key.split('_')[2].split('-')
                        defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                        all_labels.append(1.0)
                        all_defect_positions.append([defect_start, defect_end])
                    except:
                        all_labels.append(1.0)
                        all_defect_positions.append([0.0, 0.0])
            else:
                if isinstance(signal_data, dict):
                    label = signal_data.get('label', 0.0)
                    defect_start = signal_data.get('defect_start', 0.0)
                    defect_end = signal_data.get('defect_end', 0.0)
                    all_labels.append(float(label))
                    all_defect_positions.append([float(defect_start), float(defect_end)])
                else:
                    all_labels.append(0.0)
                    all_defect_positions.append([0.0, 0.0])
        except Exception as e:
            print(f"Error processing signal {scan_key}_{signal_key}: {e}")
            continue
    
    print(f"Extracted {len(all_signals)} signals from beam {beam_key}")
    return beam_key, all_signals, all_labels, all_defect_positions, all_signal_keys

def create_sequences(signals, labels, defect_positions, signal_keys, seq_length):
    """Create sequences from the signals"""
    sequences = []
    num_signals = len(signals)
    
    signal_length = len(signals[0])
    valid_signals = []
    valid_labels = []
    valid_defect_positions = []
    valid_keys = []
    
    for i, signal in enumerate(signals):
        if len(signal) == signal_length:
            valid_signals.append(signal)
            valid_labels.append(labels[i])
            valid_defect_positions.append(defect_positions[i])
            valid_keys.append(signal_keys[i])
    
    signals = valid_signals
    labels = valid_labels
    defect_positions = valid_defect_positions
    signal_keys = valid_keys
    
    num_signals = len(signals)
    print(f"Using {num_signals} valid signals with length {signal_length}")
    
    for start_idx in range(0, num_signals, seq_length):
        end_idx = min(start_idx + seq_length, num_signals)
        seq_signals = signals[start_idx:end_idx]
        seq_labels = labels[start_idx:end_idx]
        seq_defect_positions = defect_positions[start_idx:end_idx]
        seq_keys = signal_keys[start_idx:end_idx]
        
        if len(seq_signals) < 2:
            continue
        
        if len(seq_signals) < seq_length:
            pad_length = seq_length - len(seq_signals)
            seq_signals.extend([np.zeros(signal_length, dtype=np.float32) for _ in range(pad_length)])
            seq_labels.extend([0.0 for _ in range(pad_length)])
            seq_defect_positions.extend([[0.0, 0.0] for _ in range(pad_length)])
            seq_keys.extend([f"padding_{i}" for i in range(pad_length)])
        
        sequences.append({
            'signals': seq_signals,
            'labels': seq_labels,
            'defect_positions': seq_defect_positions,
            'keys': seq_keys,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    print(f"Created {len(sequences)} sequences")
    return sequences

def run_predictions(model, device, sequences):
    """Run predictions on the sequences"""
    results = []
    
    for i, seq in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)}")
        
        signals_tensor = torch.tensor(seq['signals'], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            defect_prob, defect_start, defect_end = model(signals_tensor)
        
        seq_results = []
        for j in range(min(len(seq['keys']), len(defect_prob[0]))):
            if "padding_" not in seq['keys'][j]:
                seq_results.append({
                    'key': seq['keys'][j],
                    'gt_label': seq['labels'][j],
                    'gt_position': seq['defect_positions'][j],
                    'pred_prob': defect_prob[0][j].item(),
                    'pred_start': defect_start[0][j].item(),
                    'pred_end': defect_end[0][j].item(),
                    'signal': seq['signals'][j],
                    'position_in_sequence': j
                })
        
        results.append({
            'sequence_idx': i,
            'start_idx': seq['start_idx'],
            'end_idx': seq['end_idx'],
            'signals': seq_results
        })
    
    return results

def visualize_sequence(sequence_result, save_path=None):
    """Visualize a sequence with ground truth and predictions"""
    signals = [r['signal'] for r in sequence_result['signals']]
    labels = [r['gt_label'] for r in sequence_result['signals']]
    gt_positions = [r['gt_position'] for r in sequence_result['signals']]
    pred_probs = [r['pred_prob'] for r in sequence_result['signals']]
    pred_positions = [[r['pred_start'], r['pred_end']] for r in sequence_result['signals']]
    keys = [r['key'] for r in sequence_result['signals']]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    signals_array = np.array(signals)
    im = plt.imshow(signals_array, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Signal Value')
    
    # Mark ground truth defects
    for i, (label, position) in enumerate(zip(labels, gt_positions)):
        if label > 0.5:  # If it's a defect
            start, end = position
            start_idx = int(start * signals_array.shape[1])
            end_idx = int(end * signals_array.shape[1])
            rect = Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                            fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    
    # Mark predictions
    for i, (prob, position) in enumerate(zip(pred_probs, pred_positions)):
        if prob > 0.5:  # If prediction confidence is high enough
            start, end = position
            start_idx = int(start * signals_array.shape[1])
            end_idx = int(end * signals_array.shape[1])
            rect = Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                            fill=False, edgecolor='blue', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(start_idx, i + 0.3, f"{prob:.2f}", color='blue', fontsize=8)
    
    plt.xlabel('Signal Position')
    plt.ylabel('Signal Index')
    plt.title(f'Sequence {sequence_result["sequence_idx"]} Visualization')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Ground Truth'),
        Patch(facecolor='none', edgecolor='blue', label='Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(pred_probs)), pred_probs, color='blue', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Signal Index')
    plt.ylabel('Defect Probability')
    plt.title('Prediction Probabilities')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def main():
    model, device = load_model(MODEL_PATH)
    
    beam_key, signals, labels, defect_positions, signal_keys = extract_beam_signals(JSON_FILE_PATH, BEAM_INDEX)
    
    sequences = create_sequences(signals, labels, defect_positions, signal_keys, SEQ_LENGTH)
    
    results = run_predictions(model, device, sequences)
    
    for i, result in enumerate(results):
        save_path = f"beam_{BEAM_INDEX}_sequence_{i}.png"
        visualize_sequence(result, save_path)
    
    print(f"Processed {len(results)} sequences from beam {beam_key}")
    
    total_signals = sum(len(r['signals']) for r in results)
    total_gt_defects = sum(sum(1 for s in r['signals'] if s['gt_label'] > 0.5) for r in results)
    total_pred_defects = sum(sum(1 for s in r['signals'] if s['pred_prob'] > 0.5) for r in results)
    
    print(f"Total signals: {total_signals}")
    print(f"Ground truth defects: {total_gt_defects}")
    print(f"Predicted defects: {total_pred_defects}")

if __name__ == "__main__":
    main()
