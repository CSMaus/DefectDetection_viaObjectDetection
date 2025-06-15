import os
import sys
import json
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QComboBox, QPushButton, QLabel, QFileDialog, QFrame, QGridLayout,
                            QSpinBox, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from improved_model import ImprovedMultiSignalClassifier


class PredictionWorker(QThread):
    """Worker thread for running predictions on sequences"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int, int)
    
    def __init__(self, model, json_data, seq_length=50):
        super().__init__()
        self.model = model
        self.json_data = json_data
        self.seq_length = seq_length
        self.device = next(model.parameters()).device
        
    def run(self):
        results = {}
        total_beams = len(self.json_data)
        
        for beam_idx, (beam_key, beam_data) in enumerate(self.json_data.items()):
            self.progress.emit(beam_idx, total_beams)
            
            # Extract all signals for this beam - using the same approach as json_dataset.py
            all_signals = []
            all_labels = []
            all_defect_positions = []
            all_signal_keys = []
            
            # Sort signal files by scan index (exactly as in json_dataset.py)
            signal_files = []
            for scan_key in beam_data.keys():
                scan_data = beam_data[scan_key]
                if isinstance(scan_data, dict):
                    for signal_key in scan_data.keys():
                        signal_files.append((scan_key, signal_key))
                elif isinstance(scan_data, list):
                    for i, _ in enumerate(scan_data):
                        signal_files.append((scan_key, str(i)))
            
            # Sort by scan index if possible (exactly as in json_dataset.py)
            try:
                signal_files.sort(key=lambda x: int(round(float(x[1].split('_')[0]))) 
                                if '_' in x[1] and x[1].split('_')[0].replace('.', '', 1).isdigit() else 0)
            except:
                # If sorting fails, keep original order
                pass
            
            # Process each signal in sorted order
            for scan_key, signal_key in signal_files:
                # Get signal data
                try:
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
                    
                    # Extract signal data (exactly as in json_dataset.py)
                    if isinstance(signal_data, list):
                        signal = np.array(signal_data, dtype=np.float32)
                    elif isinstance(signal_data, dict) and 'signal' in signal_data:
                        signal = np.array(signal_data['signal'], dtype=np.float32)
                    else:
                        signal = np.array(signal_data, dtype=np.float32)
                    
                    # Skip invalid signals
                    if signal is None or signal.size == 0:
                        continue
                    
                    all_signals.append(signal)
                    all_signal_keys.append(f"{scan_key}_{signal_key}")
                    
                    # Process label and defect position (exactly as in json_dataset.py)
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
                        # For list-based data, try to get label from dict or default to 0
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
                    # Skip any signals that cause errors
                    continue
            
            # Skip if no signals found
            if len(all_signals) < 2:
                continue
            
            # Create sequences from all signals (similar to json_dataset.py)
            beam_results = {}
            num_signals = len(all_signals)
            
            # Process in sequences of seq_length
            for start_idx in range(0, num_signals, self.seq_length):
                end_idx = min(start_idx + self.seq_length, num_signals)
                seq_signals = all_signals[start_idx:end_idx]
                seq_keys = all_signal_keys[start_idx:end_idx]
                seq_labels = all_labels[start_idx:end_idx]
                seq_defect_positions = all_defect_positions[start_idx:end_idx]
                
                # Skip sequences that are too short
                if len(seq_signals) < 2:
                    continue
                
                # Make sure all signals have the same length
                signal_length = len(seq_signals[0])
                valid_signals = []
                valid_keys = []
                valid_labels = []
                valid_defect_positions = []
                
                for i, signal in enumerate(seq_signals):
                    if len(signal) == signal_length:
                        valid_signals.append(signal)
                        valid_keys.append(seq_keys[i])
                        valid_labels.append(seq_labels[i])
                        valid_defect_positions.append(seq_defect_positions[i])
                
                # Skip if not enough valid signals
                if len(valid_signals) < 2:
                    continue
                
                # Use validated signals
                seq_signals = valid_signals
                seq_keys = valid_keys
                seq_labels = valid_labels
                seq_defect_positions = valid_defect_positions
                
                # Pad if needed (exactly as in json_dataset.py)
                if len(seq_signals) < self.seq_length:
                    pad_length = self.seq_length - len(seq_signals)
                    seq_signals.extend([np.zeros(signal_length, dtype=np.float32) for _ in range(pad_length)])
                    seq_labels.extend([0.0 for _ in range(pad_length)])
                    seq_defect_positions.extend([[0.0, 0.0] for _ in range(pad_length)])
                    # Pad keys with dummy values
                    seq_keys.extend([f"padding_{i}" for i in range(pad_length)])
                
                # Convert to tensor and run prediction
                try:
                    signals_tensor = torch.tensor(seq_signals, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        defect_prob, defect_start, defect_end = self.model(signals_tensor)
                    
                    # Store results for each signal in the sequence
                    for i, key in enumerate(seq_keys):
                        if i < len(defect_prob[0]) and "padding_" not in key:
                            beam_results[key] = {
                                'gt_label': seq_labels[i],
                                'gt_position': seq_defect_positions[i],
                                'pred_prob': defect_prob[0][i].item(),
                                'pred_start': defect_start[0][i].item(),
                                'pred_end': defect_end[0][i].item(),
                                'signal': seq_signals[i].tolist(),
                                'sequence_idx': start_idx // self.seq_length,
                                'position_in_sequence': i
                            }
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                    continue
            
            # Store results for this beam
            if beam_results:
                results[beam_key] = beam_results
        
        self.finished.emit(results)


class ModelTesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Improved Model Tester")
        self.setGeometry(100, 100, 1200, 800)
        
        # Model and data attributes
        self.model = None
        self.json_dir = None
        self.json_files = []
        self.current_json_data = None
        self.prediction_results = {}
        self.beam_keys = []
        self.scan_keys = []
        self.signal_keys = []
        self.sequence_indices = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup UI
        self.setup_ui()
        
        # Load model
        self.load_model()
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Controls area
        controls_frame = QFrame()
        controls_layout = QGridLayout(controls_frame)
        
        # JSON directory selection
        self.dir_label = QLabel("JSON Directory:")
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_json_dir)
        
        # JSON file selection
        self.file_label = QLabel("JSON File:")
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.load_json_file)
        
        # Model selection
        self.model_label = QLabel("Model:")
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.load_model)
        self.model_path_label = QLabel("No model loaded")
        
        # Run predictions button
        self.predict_button = QPushButton("Run Predictions")
        self.predict_button.clicked.connect(self.run_predictions)
        self.predict_status = QLabel("Ready")
        
        # Beam selection
        self.beam_label = QLabel("Beam:")
        self.beam_combo = QComboBox()
        self.beam_combo.currentIndexChanged.connect(self.update_scan_combo)
        
        # Sequence selection
        self.sequence_label = QLabel("Sequence:")
        self.sequence_combo = QComboBox()
        self.sequence_combo.currentIndexChanged.connect(self.update_sequence_visualization)
        
        # Signal selection
        self.signal_label = QLabel("Signal:")
        self.signal_combo = QComboBox()
        self.signal_combo.currentIndexChanged.connect(self.update_signal_visualization)
        
        # Add controls to layout
        controls_layout.addWidget(self.dir_label, 0, 0)
        controls_layout.addWidget(self.dir_button, 0, 1)
        controls_layout.addWidget(self.file_label, 0, 2)
        controls_layout.addWidget(self.file_combo, 0, 3)
        controls_layout.addWidget(self.model_label, 1, 0)
        controls_layout.addWidget(self.model_button, 1, 1)
        controls_layout.addWidget(self.model_path_label, 1, 2, 1, 2)
        controls_layout.addWidget(self.predict_button, 2, 0)
        controls_layout.addWidget(self.predict_status, 2, 1, 1, 3)
        controls_layout.addWidget(self.beam_label, 3, 0)
        controls_layout.addWidget(self.beam_combo, 3, 1)
        controls_layout.addWidget(self.sequence_label, 3, 2)
        controls_layout.addWidget(self.sequence_combo, 3, 3)
        controls_layout.addWidget(self.signal_label, 4, 0)
        controls_layout.addWidget(self.signal_combo, 4, 1, 1, 3)
        
        # Tab widget for different visualizations
        self.tab_widget = QTabWidget()
        
        # Sequence visualization tab
        self.sequence_tab = QWidget()
        sequence_layout = QVBoxLayout(self.sequence_tab)
        self.sequence_figure = Figure(figsize=(10, 6))
        self.sequence_canvas = FigureCanvas(self.sequence_figure)
        sequence_layout.addWidget(self.sequence_canvas)
        
        # Signal visualization tab
        self.signal_tab = QWidget()
        signal_layout = QVBoxLayout(self.signal_tab)
        self.signal_figure = Figure(figsize=(10, 6))
        self.signal_canvas = FigureCanvas(self.signal_figure)
        signal_layout.addWidget(self.signal_canvas)
        
        # Add tabs
        self.tab_widget.addTab(self.sequence_tab, "Sequence View")
        self.tab_widget.addTab(self.signal_tab, "Signal View")
        
        # Add widgets to main layout
        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.tab_widget)
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def load_model(self):
        """Load the trained model"""
        model_path = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth)")[0]
        if not model_path:
            return
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model instance
            signal_length = 320  # Default, will be updated when processing data
            hidden_sizes = [128, 64, 32]
            num_heads = 8
            num_transformer_layers = 4
            
            self.model = ImprovedMultiSignalClassifier(
                signal_length=signal_length,
                hidden_sizes=hidden_sizes,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Update model path label
            self.model_path_label.setText(os.path.basename(model_path))
            
            print(f"Model loaded from {model_path}")
        except Exception as e:
            self.show_error(f"Error loading model: {e}")
    
    def select_json_dir(self):
        """Select directory containing JSON files"""
        directory = QFileDialog.getExistingDirectory(self, "Select JSON Directory")
        if not directory:
            return
        
        self.json_dir = directory
        self.json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
        # Update JSON file combo box
        self.file_combo.clear()
        self.file_combo.addItems(self.json_files)
    
    def load_json_file(self, index):
        """Load selected JSON file"""
        if index < 0 or not self.json_dir:
            return
        
        json_file = self.json_files[index]
        json_path = os.path.join(self.json_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                self.current_json_data = json.load(f)
            
            # Update beam combo box
            self.beam_keys = list(self.current_json_data.keys())
            self.beam_combo.clear()
            self.beam_combo.addItems(self.beam_keys)
            
            # Clear prediction results
            self.prediction_results = {}
            self.predict_status.setText("Ready - JSON file loaded")
        except Exception as e:
            self.show_error(f"Error loading JSON file: {e}")
    
    def run_predictions(self):
        """Run predictions on all sequences in the JSON file"""
        if not self.model:
            self.show_error("No model loaded")
            return
        
        if not self.current_json_data:
            self.show_error("No JSON file loaded")
            return
        
        self.predict_status.setText("Running predictions...")
        self.predict_button.setEnabled(False)
        
        # Create worker thread
        self.worker = PredictionWorker(self.model, self.current_json_data)
        self.worker.finished.connect(self.predictions_finished)
        self.worker.progress.connect(self.update_prediction_progress)
        self.worker.start()
    
    def update_prediction_progress(self, current, total):
        """Update prediction progress"""
        self.predict_status.setText(f"Processing beam {current+1}/{total}...")
    
    def predictions_finished(self, results):
        """Handle prediction results"""
        self.prediction_results = results
        self.predict_status.setText(f"Predictions complete for {len(results)} beams")
        self.predict_button.setEnabled(True)
        
        # Update visualization
        if self.beam_combo.currentIndex() >= 0:
            self.update_scan_combo(self.beam_combo.currentIndex())
    
    def update_scan_combo(self, index):
        """Update sequence combo box based on selected beam"""
        if index < 0 or not self.beam_keys:
            return
        
        beam_key = self.beam_keys[index]
        
        if beam_key in self.prediction_results:
            # Get unique sequence indices from the prediction results
            sequence_indices = set()
            for key, result in self.prediction_results[beam_key].items():
                if 'sequence_idx' in result:
                    sequence_indices.add(result['sequence_idx'])
            
            self.sequence_indices = sorted(list(sequence_indices))
            self.sequence_combo.clear()
            self.sequence_combo.addItems([f"Sequence {idx}" for idx in self.sequence_indices])
        else:
            self.sequence_combo.clear()
            self.sequence_indices = []
    
    def update_sequence_visualization(self, index):
        """Update visualization for selected sequence"""
        if index < 0 or not self.sequence_indices or self.beam_combo.currentIndex() < 0:
            return
        
        beam_key = self.beam_keys[self.beam_combo.currentIndex()]
        sequence_idx = self.sequence_indices[index]
        
        if beam_key in self.prediction_results:
            # Get signals for this sequence
            sequence_signals = []
            sequence_labels = []
            sequence_gt_positions = []
            sequence_pred_probs = []
            sequence_pred_positions = []
            signal_keys = []
            
            for key, result in self.prediction_results[beam_key].items():
                if result.get('sequence_idx') == sequence_idx:
                    pos = result.get('position_in_sequence', 0)
                    # Ensure we have enough space in our lists
                    while len(sequence_signals) <= pos:
                        sequence_signals.append(None)
                        sequence_labels.append(None)
                        sequence_gt_positions.append(None)
                        sequence_pred_probs.append(None)
                        sequence_pred_positions.append(None)
                        signal_keys.append(None)
                    
                    sequence_signals[pos] = result['signal']
                    sequence_labels[pos] = result['gt_label']
                    sequence_gt_positions[pos] = result['gt_position']
                    sequence_pred_probs[pos] = result['pred_prob']
                    sequence_pred_positions[pos] = [result['pred_start'], result['pred_end']]
                    signal_keys[pos] = key
            
            # Remove None values
            valid_indices = [i for i, x in enumerate(sequence_signals) if x is not None]
            sequence_signals = [sequence_signals[i] for i in valid_indices]
            sequence_labels = [sequence_labels[i] for i in valid_indices]
            sequence_gt_positions = [sequence_gt_positions[i] for i in valid_indices]
            sequence_pred_probs = [sequence_pred_probs[i] for i in valid_indices]
            sequence_pred_positions = [sequence_pred_positions[i] for i in valid_indices]
            signal_keys = [signal_keys[i] for i in valid_indices]
            
            # Update signal combo box
            self.signal_keys = signal_keys
            self.signal_combo.clear()
            self.signal_combo.addItems(signal_keys)
            
            # Visualize sequence
            self.visualize_sequence(
                sequence_signals, 
                sequence_labels, 
                sequence_gt_positions, 
                sequence_pred_probs, 
                sequence_pred_positions
            )
    
    def update_signal_visualization(self, index):
        """Update visualization for selected signal"""
        if index < 0 or not self.signal_keys or self.beam_combo.currentIndex() < 0 or self.sequence_combo.currentIndex() < 0:
            return
        
        beam_key = self.beam_keys[self.beam_combo.currentIndex()]
        signal_key = self.signal_keys[index]
        
        if beam_key in self.prediction_results and signal_key in self.prediction_results[beam_key]:
            result = self.prediction_results[beam_key][signal_key]
            self.visualize_signal(result)
    
    def visualize_sequence(self, signals, labels, gt_positions, pred_probs, pred_positions):
        """Visualize a sequence of signals with ground truth and predictions"""
        # Clear figure
        self.sequence_figure.clear()
        
        # Create a heatmap-style visualization
        ax = self.sequence_figure.add_subplot(111)
        
        # Convert signals to a 2D array
        signals_array = np.array(signals)
        
        # Display as an image
        im = ax.imshow(signals_array, aspect='auto', cmap='viridis')
        
        # Add colorbar
        self.sequence_figure.colorbar(im, ax=ax, label='Signal Value')
        
        # Mark ground truth defects
        for i, (label, position) in enumerate(zip(labels, gt_positions)):
            if label > 0.5:  # If it's a defect
                start, end = position
                start_idx = int(start * signals_array.shape[1])
                end_idx = int(end * signals_array.shape[1])
                rect = plt.Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                                    fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        
        # Mark predictions
        for i, (prob, position) in enumerate(zip(pred_probs, pred_positions)):
            if prob > 0.5:  # If prediction confidence is high enough
                start, end = position
                start_idx = int(start * signals_array.shape[1])
                end_idx = int(end * signals_array.shape[1])
                rect = plt.Rectangle((start_idx, i - 0.5), end_idx - start_idx, 1, 
                                    fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(rect)
                ax.text(start_idx, i + 0.3, f"{prob:.2f}", color='blue', fontsize=8)
        
        # Add labels and title
        ax.set_xlabel('Signal Position')
        ax.set_ylabel('Signal Index')
        ax.set_title(f'Sequence Visualization - Beam: {self.beam_keys[self.beam_combo.currentIndex()]}, ' +
                    f'Sequence: {self.sequence_indices[self.sequence_combo.currentIndex()]}')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='red', label='Ground Truth'),
            Patch(facecolor='none', edgecolor='blue', label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Update canvas
        self.sequence_canvas.draw()
    
    def visualize_signal(self, result):
        """Visualize a single signal with ground truth and prediction"""
        # Clear figure
        self.signal_figure.clear()
        
        # Create plot
        ax = self.signal_figure.add_subplot(111)
        
        # Plot signal
        signal = np.array(result['signal'])
        ax.plot(signal, 'b-', alpha=0.7)
        
        # Plot ground truth
        if result['gt_label'] > 0.5:
            start_idx = int(result['gt_position'][0] * len(signal))
            end_idx = int(result['gt_position'][1] * len(signal))
            ax.axvspan(start_idx, end_idx, alpha=0.3, color='green', label='Ground Truth')
        
        # Plot prediction
        if result['pred_prob'] > 0.5:
            start_idx = int(result['pred_start'] * len(signal))
            end_idx = int(result['pred_end'] * len(signal))
            ax.axvspan(start_idx, end_idx, alpha=0.3, color='red', label='Prediction')
        
        # Add title and legend
        ax.set_title(f"Signal: {self.signal_keys[self.signal_combo.currentIndex()]}\nPrediction: {result['pred_prob']:.4f}")
        if (result['gt_label'] > 0.5) or (result['pred_prob'] > 0.5):
            ax.legend()
        
        # Add grid
        ax.grid(True)
        
        # Update canvas
        self.signal_canvas.draw()
    
    def show_error(self, message):
        """Show error message"""
        self.predict_status.setText(f"Error: {message}")
        print(f"Error: {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelTesterApp()
    window.show()
    sys.exit(app.exec())
