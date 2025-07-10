import os
import json
import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

from improved_model import ImprovedMultiSignalClassifier


class ModelEvaluator:
    def __init__(self, model_path, json_dir, seq_length=50, device=None):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to the trained model checkpoint
            json_dir: Directory containing JSON files with ground truth
            seq_length: Sequence length used during training
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.json_dir = json_dir
        self.seq_length = seq_length
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
        # Storage for all predictions and ground truth
        self.all_predictions = {
            'defect_probs': [],
            'defect_starts': [],
            'defect_ends': [],
            'gt_labels': [],
            'gt_starts': [],
            'gt_ends': []
        }
        
        print(f"Model loaded on device: {self.device}")
    
    def _load_model(self):
        """Load the trained model from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model parameters from checkpoint if available
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Initialize model with default parameters (you may need to adjust these)
            model = ImprovedMultiSignalClassifier(
                signal_length=320,  # Adjust if different
                hidden_sizes=[128, 64, 32],  # Adjust if different
                num_heads=8,
                dropout=0.2,
                num_transformer_layers=4
            )
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_sequences_from_json(self, json_file_path):
        """Load sequences from a single JSON file (same logic as in dataset)"""
        sequences = []
        labels = []
        defect_positions = []
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Process each beam
            for beam_key in data.keys():
                beam_data = data[beam_key]
                
                # Sort scan keys by index
                scans_keys_sorted = sorted(beam_data.keys(), key=lambda x: int(x.split('_')[0]))
                
                # Skip if not enough scans for a full sequence
                if len(scans_keys_sorted) < self.seq_length:
                    continue
                
                # Extract all signals, labels, and defect positions for this beam
                all_scans_for_beam = {}
                all_lbls_for_beam = {}
                all_defects_for_beam = {}
                
                scan_idx = 0
                for scan_key in scans_keys_sorted:
                    scan_data = beam_data[scan_key]
                    all_scans_for_beam[str(scan_idx)] = scan_data
                    
                    # Extract label and defect position
                    if scan_key.split('_')[1] == "Health":
                        all_lbls_for_beam[str(scan_idx)] = 0
                        all_defects_for_beam[str(scan_idx)] = [None, None]
                    else:
                        all_lbls_for_beam[str(scan_idx)] = 1
                        try:
                            defect_range = scan_key.split('_')[2].split('-')
                            defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                            all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]
                        except:
                            all_defects_for_beam[str(scan_idx)] = [0.0, 0.0]
                    
                    scan_idx += 1
                
                # Create sequences from this beam
                num_of_seqs_for_beam = math.ceil(len(scans_keys_sorted) / self.seq_length)
                
                for i in range(num_of_seqs_for_beam):
                    sequence = []
                    seq_labels = []
                    seq_defects = []
                    
                    # Determine start and end indices for this sequence
                    if i < num_of_seqs_for_beam - 1:
                        start_idx = i * self.seq_length
                        end_idx = start_idx + self.seq_length
                    else:
                        start_idx = len(scans_keys_sorted) - self.seq_length
                        end_idx = len(scans_keys_sorted)
                    
                    if start_idx < 0:
                        continue
                    
                    # Extract signals for this sequence
                    for j in range(start_idx, end_idx):
                        try:
                            scan_data = all_scans_for_beam[str(j)]
                            
                            # Convert scan data to numpy array
                            if isinstance(scan_data, list):
                                signal = np.array(scan_data, dtype=np.float32)
                            elif isinstance(scan_data, dict) and 'signal' in scan_data:
                                signal = np.array(scan_data['signal'], dtype=np.float32)
                            else:
                                signal = np.array(scan_data, dtype=np.float32)
                            
                            sequence.append(signal)
                            seq_labels.append(all_lbls_for_beam[str(j)])
                            seq_defects.append(all_defects_for_beam[str(j)])
                        except Exception as e:
                            print(f"Error processing scan {j} in beam {beam_key}: {e}")
                            continue
                    
                    # Skip if sequence doesn't have exactly seq_length signals
                    if len(sequence) != self.seq_length:
                        continue
                    
                    # Ensure all signals have the same length
                    signal_length = len(sequence[0])
                    valid = True
                    for signal in sequence:
                        if len(signal) != signal_length:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Format defects
                    formatted_defects = []
                    for defect in seq_defects:
                        if defect[0] is None:
                            formatted_defects.append([0.0, 0.0])
                        else:
                            formatted_defects.append([float(defect[0]), float(defect[1])])
                    
                    sequences.append(np.array(sequence, dtype=np.float32))
                    labels.append(np.array(seq_labels, dtype=np.float32))
                    defect_positions.append(np.array(formatted_defects, dtype=np.float32))
        
        except Exception as e:
            print(f"Error loading {json_file_path}: {e}")
        
        return sequences, labels, defect_positions
    
    def run_evaluation(self):
        """Run evaluation on all JSON files in the directory"""
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files for evaluation")
        
        total_sequences = 0
        
        # Process each JSON file
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            json_path = os.path.join(self.json_dir, json_file)
            sequences, labels, defect_positions = self._load_sequences_from_json(json_path)
            
            # Run predictions on each sequence
            for seq_idx, (sequence, label, defect_pos) in enumerate(zip(sequences, labels, defect_positions)):
                try:
                    # Convert to tensor
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Run prediction
                    with torch.no_grad():
                        defect_prob, defect_start, defect_end = self.model(sequence_tensor)
                    
                    # Store predictions and ground truth
                    self.all_predictions['defect_probs'].extend(defect_prob[0].cpu().numpy())
                    self.all_predictions['defect_starts'].extend(defect_start[0].cpu().numpy())
                    self.all_predictions['defect_ends'].extend(defect_end[0].cpu().numpy())
                    self.all_predictions['gt_labels'].extend(label)
                    self.all_predictions['gt_starts'].extend(defect_pos[:, 0])
                    self.all_predictions['gt_ends'].extend(defect_pos[:, 1])
                    
                    total_sequences += 1
                    
                except Exception as e:
                    print(f"Error processing sequence {seq_idx} from {json_file}: {e}")
                    continue
        
        print(f"Processed {total_sequences} sequences total")
        return self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate comprehensive metrics"""
        # Convert to numpy arrays
        pred_probs = np.array(self.all_predictions['defect_probs'])
        pred_starts = np.array(self.all_predictions['defect_starts'])
        pred_ends = np.array(self.all_predictions['defect_ends'])
        gt_labels = np.array(self.all_predictions['gt_labels'])
        gt_starts = np.array(self.all_predictions['gt_starts'])
        gt_ends = np.array(self.all_predictions['gt_ends'])
        
        # Binary predictions for classification metrics
        pred_labels = (pred_probs > 0.5).astype(int)
        gt_labels_binary = (gt_labels > 0.5).astype(int)
        
        metrics = {}
        
        # === DEFECT DETECTION METRICS ===
        metrics['detection'] = {
            'accuracy': accuracy_score(gt_labels_binary, pred_labels),
            'precision': precision_score(gt_labels_binary, pred_labels, zero_division=0),
            'recall': recall_score(gt_labels_binary, pred_labels, zero_division=0),
            'f1_score': f1_score(gt_labels_binary, pred_labels, zero_division=0),
            'auc_roc': roc_auc_score(gt_labels_binary, pred_probs) if len(np.unique(gt_labels_binary)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(gt_labels_binary, pred_labels)
        metrics['detection']['confusion_matrix'] = cm.tolist()
        
        # === POSITION PREDICTION METRICS (only for defective signals) ===
        defect_mask = gt_labels_binary == 1
        
        if np.sum(defect_mask) > 0:
            # Filter to only defective signals
            pred_starts_defect = pred_starts[defect_mask]
            pred_ends_defect = pred_ends[defect_mask]
            gt_starts_defect = gt_starts[defect_mask]
            gt_ends_defect = gt_ends[defect_mask]
            
            # Position prediction metrics
            start_mae = np.mean(np.abs(pred_starts_defect - gt_starts_defect))
            end_mae = np.mean(np.abs(pred_ends_defect - gt_ends_defect))
            start_rmse = np.sqrt(np.mean((pred_starts_defect - gt_starts_defect) ** 2))
            end_rmse = np.sqrt(np.mean((pred_ends_defect - gt_ends_defect) ** 2))
            
            # IoU-like metric for position overlap
            pred_lengths = np.abs(pred_ends_defect - pred_starts_defect)
            gt_lengths = np.abs(gt_ends_defect - gt_starts_defect)
            
            # Calculate overlap
            overlap_starts = np.maximum(pred_starts_defect, gt_starts_defect)
            overlap_ends = np.minimum(pred_ends_defect, gt_ends_defect)
            overlaps = np.maximum(0, overlap_ends - overlap_starts)
            unions = pred_lengths + gt_lengths - overlaps
            ious = overlaps / (unions + 1e-8)
            
            metrics['position'] = {
                'start_mae': float(start_mae),
                'end_mae': float(end_mae),
                'start_rmse': float(start_rmse),
                'end_rmse': float(end_rmse),
                'mean_iou': float(np.mean(ious)),
                'median_iou': float(np.median(ious)),
                'iou_std': float(np.std(ious)),
                'num_defective_samples': int(np.sum(defect_mask))
            }
            
            # Position accuracy at different thresholds
            thresholds = [0.1, 0.2, 0.3, 0.5]
            position_accuracies = {}
            for thresh in thresholds:
                accurate_positions = np.sum(ious >= thresh)
                position_accuracies[f'accuracy_at_iou_{thresh}'] = float(accurate_positions / len(ious))
            
            metrics['position']['position_accuracies'] = position_accuracies
        else:
            metrics['position'] = {
                'start_mae': 0.0,
                'end_mae': 0.0,
                'start_rmse': 0.0,
                'end_rmse': 0.0,
                'mean_iou': 0.0,
                'median_iou': 0.0,
                'iou_std': 0.0,
                'num_defective_samples': 0,
                'position_accuracies': {}
            }
        
        # === OVERALL STATISTICS ===
        metrics['overall'] = {
            'total_samples': len(gt_labels),
            'defective_samples': int(np.sum(gt_labels_binary)),
            'healthy_samples': int(np.sum(gt_labels_binary == 0)),
            'defect_ratio': float(np.mean(gt_labels_binary)),
            'mean_pred_confidence': float(np.mean(pred_probs)),
            'std_pred_confidence': float(np.std(pred_probs))
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print formatted metrics"""
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total samples: {metrics['overall']['total_samples']:,}")
        print(f"  Defective samples: {metrics['overall']['defective_samples']:,}")
        print(f"  Healthy samples: {metrics['overall']['healthy_samples']:,}")
        print(f"  Defect ratio: {metrics['overall']['defect_ratio']:.3f}")
        print(f"  Mean prediction confidence: {metrics['overall']['mean_pred_confidence']:.3f}")
        
        # Detection metrics
        print(f"\nDEFECT DETECTION METRICS:")
        print(f"  Accuracy: {metrics['detection']['accuracy']:.4f}")
        print(f"  Precision: {metrics['detection']['precision']:.4f}")
        print(f"  Recall: {metrics['detection']['recall']:.4f}")
        print(f"  F1-Score: {metrics['detection']['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['detection']['auc_roc']:.4f}")
        
        # Confusion matrix
        cm = np.array(metrics['detection']['confusion_matrix'])
        print(f"\n  Confusion Matrix:")
        print(f"    True Negative: {cm[0,0]:,}, False Positive: {cm[0,1]:,}")
        print(f"    False Negative: {cm[1,0]:,}, True Positive: {cm[1,1]:,}")
        
        # Position metrics
        print(f"\nPOSITION PREDICTION METRICS:")
        if metrics['position']['num_defective_samples'] > 0:
            print(f"  Number of defective samples: {metrics['position']['num_defective_samples']:,}")
            print(f"  Start position MAE: {metrics['position']['start_mae']:.4f}")
            print(f"  End position MAE: {metrics['position']['end_mae']:.4f}")
            print(f"  Start position RMSE: {metrics['position']['start_rmse']:.4f}")
            print(f"  End position RMSE: {metrics['position']['end_rmse']:.4f}")
            print(f"  Mean IoU: {metrics['position']['mean_iou']:.4f}")
            print(f"  Median IoU: {metrics['position']['median_iou']:.4f}")
            
            print(f"\n  Position Accuracy at IoU thresholds:")
            for key, value in metrics['position']['position_accuracies'].items():
                threshold = key.split('_')[-1]
                print(f"    IoU >= {threshold}: {value:.4f}")
        else:
            print("  No defective samples found for position evaluation")
        
        print("\n" + "="*80)
    
    def save_metrics(self, metrics, output_path):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {output_path}")
    
    def plot_metrics(self, metrics, save_dir=None):
        """Create visualization plots"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['detection']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Healthy', 'Defective'], 
                   yticklabels=['Healthy', 'Defective'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Metrics Summary
        plt.figure(figsize=(12, 8))
        
        # Detection metrics
        detection_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        detection_values = [metrics['detection'][m] for m in detection_metrics]
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(detection_metrics, detection_values, color='skyblue', alpha=0.7)
        plt.title('Detection Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, detection_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Position metrics (if available)
        if metrics['position']['num_defective_samples'] > 0:
            plt.subplot(2, 2, 2)
            pos_metrics = ['start_mae', 'end_mae', 'start_rmse', 'end_rmse']
            pos_values = [metrics['position'][m] for m in pos_metrics]
            bars = plt.bar(pos_metrics, pos_values, color='lightcoral', alpha=0.7)
            plt.title('Position Error Metrics')
            plt.ylabel('Error')
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, pos_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pos_values)*0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.subplot(2, 2, 3)
            iou_metrics = list(metrics['position']['position_accuracies'].keys())
            iou_values = list(metrics['position']['position_accuracies'].values())
            bars = plt.bar([m.split('_')[-1] for m in iou_metrics], iou_values, color='lightgreen', alpha=0.7)
            plt.title('Position Accuracy at IoU Thresholds')
            plt.ylabel('Accuracy')
            plt.xlabel('IoU Threshold')
            
            for bar, value in zip(bars, iou_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on JSON dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length used during training')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.json_dir, args.seq_length)
    
    # Run evaluation
    print("Starting model evaluation...")
    metrics = evaluator.run_evaluation()
    
    # Print results
    evaluator.print_metrics(metrics)
    
    # Save results
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    evaluator.save_metrics(metrics, metrics_path)
    
    # Create plots
    evaluator.plot_metrics(metrics, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    # Set your paths here for PyCharm execution
    MODEL_PATH = "models/improved_model_20250710_193851/best_model.pth"  # CHANGE THIS to your actual model path
    JSON_DIR = "json_data"    # CHANGE THIS if your JSON directory is different
    OUTPUT_DIR = "evaluation_results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run evaluation directly (for PyCharm)
    print("Starting model evaluation...")
    evaluator = ModelEvaluator(MODEL_PATH, JSON_DIR)
    metrics = evaluator.run_evaluation()
    evaluator.print_metrics(metrics)
    evaluator.save_metrics(metrics, os.path.join(OUTPUT_DIR, 'evaluation_metrics.json'))
    evaluator.plot_metrics(metrics, OUTPUT_DIR)
    
    print(f"\nEvaluation complete! Results saved to: {OUTPUT_DIR}")
