import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import math
from tabulate import tabulate

from improved_model import ImprovedMultiSignalClassifier

# JSON_FILE_PATH = 'json_data/WOT-D456_A4_003_Ch-0_D0.5-10.json'
JSON_FILE_PATH = 'json_data/WOT-D456_A4_001_Ch-0_D0.5-21.json'
BEAM_INDEX = 28
SEQUENCE_INDEX = 0
MODEL_PATH = 'models/improved_model_20250615_193609/best_model.pth'
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
    """Create sequences for each beam index in json file,
    and make for them ground truth labels and defect positions if label is 1"""
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

        scan_idx = 0
        for scan_key in scans_keys_sorted:
            scan_data = data[beam_key][scan_key]
            # scan_idx = int(scan_key.split('_')[0])

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

            scan_idx += 1

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

def run_predictions_for_sequence(model, device, sequence, labels, defects, beam_idx, seq_idx):
    """Run predictions on a specific sequence and print detailed results"""
    print(f"\nProcessing Beam {beam_idx}, Sequence {seq_idx}")
    
    # Convert sequence to tensor
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        defect_prob, defect_start, defect_end = model(sequence_tensor)
    
    results = []
    for i in range(len(sequence)):
        if i < len(defect_prob[0]):
            gt_label = labels[i]
            gt_defect = defects[i]
            
            pred_prob = defect_prob[0][i].item()
            pred_label = 1 if pred_prob > 0.5 else 0
            pred_start = defect_start[0][i].item()
            pred_end = defect_end[0][i].item()
            
            # Format defect positions for display
            if gt_defect[0] is None:
                gt_defect_str = "None"
            else:
                gt_defect_str = f"{gt_defect[0]:.2f}-{gt_defect[1]:.2f}"
            
            pred_defect_str = f"{pred_start:.2f}-{pred_end:.2f}" if pred_label == 1 else "None"
            
            # Check if prediction matches ground truth
            label_match = "✓" if pred_label == gt_label else "✗"
            
            results.append([
                i,  # Signal index
                gt_label,  # Ground truth label
                gt_defect_str,  # Ground truth defect position
                f"{pred_prob:.4f}",  # Prediction probability
                pred_label,  # Predicted label
                pred_defect_str,  # Predicted defect position
                label_match  # Match indicator
            ])
    
    # Print results in a table
    headers = ["Signal", "GT Label", "GT Defect", "Pred Prob", "Pred Label", "Pred Defect", "Match"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Calculate accuracy metrics
    correct_labels = sum(1 for r in results if r[1] == r[4])
    accuracy = correct_labels / len(results) if results else 0
    
    true_positives = sum(1 for r in results if r[1] == 1 and r[4] == 1)
    false_positives = sum(1 for r in results if r[1] == 0 and r[4] == 1)
    false_negatives = sum(1 for r in results if r[1] == 1 and r[4] == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'sequence': sequence,
        'labels': labels,
        'defects': defects,
        'pred_probs': [r[3] for r in results],
        'pred_labels': [r[4] for r in results],
        'pred_defects': [r[5] for r in results],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def visualize_sequence_results(results, beam_idx, seq_idx, save_path=None):
    """Visualize sequence results with ground truth and predictions"""
    sequence = results['sequence']
    labels = results['labels']
    defects = results['defects']
    pred_probs = [float(p) for p in results['pred_probs']]
    
    plt.figure(figsize=(15, 10))
    
    # Plot sequence heatmap
    plt.subplot(2, 1, 1)
    signals_array = np.array(sequence)
    im = plt.imshow(signals_array, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Signal Value')
    
    # Mark ground truth defects
    for i, (label, defect) in enumerate(zip(labels, defects)):
        if label > 0.5:  # If it's a defect
            start, end = defect
            if start is not None and end is not None:
                start_idx = int(start * signals_array.shape[1])
                end_idx = int(end * signals_array.shape[1])
                rect = Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                                fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
    
    # Mark predictions
    for i, prob in enumerate(pred_probs):
        if float(prob) > 0.5:  # If prediction confidence is high enough
            # Parse the defect position string
            defect_str = results['pred_defects'][i]
            if defect_str != "None":
                try:
                    start, end = map(float, defect_str.split('-'))
                    start_idx = int(start * signals_array.shape[1])
                    end_idx = int(end * signals_array.shape[1])
                    rect = Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                                    fill=False, edgecolor='blue', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(start_idx, i + 0.3, f"{float(prob):.2f}", color='blue', fontsize=8)
                except:
                    pass
    
    plt.xlabel('Signal Position')
    plt.ylabel('Signal Index')
    plt.title(f'Beam {beam_idx}, Sequence {seq_idx} Visualization')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Ground Truth'),
        Patch(facecolor='none', edgecolor='blue', label='Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    plt.bar(range(len(pred_probs)), [float(p) for p in pred_probs], color='blue', alpha=0.7)
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
    # Load model
    model, device = load_model(MODEL_PATH)
    
    # Load sequences using your function
    print(f"Loading sequences from {JSON_FILE_PATH}")
    sequences_by_beams, labels_by_beams, defects_by_beams = load_sequences(JSON_FILE_PATH, SEQ_LENGTH)
    
    # Check if the requested beam and sequence exist
    beam_idx_str = str(BEAM_INDEX)
    seq_idx_str = str(SEQUENCE_INDEX)
    
    if beam_idx_str not in sequences_by_beams:
        available_beams = list(sequences_by_beams.keys())
        print(f"Beam index {BEAM_INDEX} not found. Available beam indices: {available_beams}")
        return
    
    if seq_idx_str not in sequences_by_beams[beam_idx_str]:
        available_seqs = list(sequences_by_beams[beam_idx_str].keys())
        print(f"Sequence index {SEQUENCE_INDEX} not found for beam {BEAM_INDEX}. Available sequence indices: {available_seqs}")
        return
    
    # Get the requested sequence, labels, and defects
    sequence = sequences_by_beams[beam_idx_str][seq_idx_str]
    labels = labels_by_beams[beam_idx_str][seq_idx_str]
    defects = defects_by_beams[beam_idx_str][seq_idx_str]
    
    print(f"Selected Beam {BEAM_INDEX}, Sequence {SEQUENCE_INDEX}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Number of defects in ground truth: {sum(1 for l in labels if l > 0)}")
    
    # Run predictions and print detailed results
    results = run_predictions_for_sequence(model, device, sequence, labels, defects, BEAM_INDEX, SEQUENCE_INDEX)
    
    # Visualize the results
    save_path = f"beam_{BEAM_INDEX}_sequence_{SEQUENCE_INDEX}.png"
    visualize_sequence_results(results, BEAM_INDEX, SEQUENCE_INDEX, save_path)
    
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
