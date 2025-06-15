import os
import sys
import json
import torch
import numpy as np
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QComboBox, QPushButton, QLabel, QFileDialog, QFrame, QGridLayout,
                            QSpinBox, QSplitter, QTabWidget, QSlider)
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
        
        # Process each beam in the JSON data
        for beam_idx, (beam_key, beam_data) in enumerate(self.json_data.items()):
            self.progress.emit(beam_idx, total_beams)
            
            # Get sequences, labels, and defects for this beam using the load_sequences approach
            sequences_by_beams, labels_by_beams, defects_by_beams = self.load_sequences_for_beam(beam_key, beam_data)
            
            # Skip if no sequences were created for this beam
            if not sequences_by_beams:
                continue
            
            # Process each sequence for this beam
            beam_results = {}
            beam_idx_str = beam_key.split('_')[1]  # Extract beam index from beam_key
            
            for seq_idx, sequence in sequences_by_beams[beam_idx_str].items():
                labels = labels_by_beams[beam_idx_str][seq_idx]
                defects = defects_by_beams[beam_idx_str][seq_idx]
                
                # Skip if sequence is too short
                if len(sequence) < self.seq_length:
                    continue
                
                # Convert sequence to tensor for prediction
                try:
                    # Convert sequence to numpy array
                    sequence_array = []
                    for signal in sequence:
                        if isinstance(signal, list):
                            sequence_array.append(np.array(signal, dtype=np.float32))
                        elif isinstance(signal, dict) and 'signal' in signal:
                            sequence_array.append(np.array(signal['signal'], dtype=np.float32))
                        else:
                            try:
                                sequence_array.append(np.array(signal, dtype=np.float32))
                            except:
                                # Skip invalid signals
                                continue
                    
                    # Skip if not enough valid signals
                    if len(sequence_array) < self.seq_length:
                        continue
                    
                    # Convert to tensor
                    sequence_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Run prediction
                    with torch.no_grad():
                        defect_prob, defect_start, defect_end = self.model(sequence_tensor)
                    
                    # Store results for each signal in the sequence
                    for i in range(len(sequence)):
                        if i < len(defect_prob[0]):
                            # Create a unique key for this signal
                            signal_key = f"{beam_key}_seq{seq_idx}_sig{i}"
                            
                            # Format defect position
                            if defects[i][0] is None:
                                gt_position = [0.0, 0.0]
                            else:
                                gt_position = [float(defects[i][0]), float(defects[i][1])]
                            
                            # Store result
                            beam_results[signal_key] = {
                                'gt_label': float(labels[i]),
                                'gt_position': gt_position,
                                'pred_prob': defect_prob[0][i].item(),
                                'pred_start': defect_start[0][i].item(),
                                'pred_end': defect_end[0][i].item(),
                                'signal': sequence_array[i].tolist(),
                                'sequence_idx': int(seq_idx),
                                'position_in_sequence': i
                            }
                except Exception as e:
                    print(f"Error processing sequence {seq_idx} in beam {beam_key}: {e}")
                    continue
            
            # Store results for this beam
            if beam_results:
                results[beam_key] = beam_results
        
        self.finished.emit(results)
    
    def load_sequences_for_beam(self, beam_key, beam_data):
        """Create sequences for a beam using the same approach as load_sequences function"""
        sequences_by_beams = {}
        labels_by_beams = {}
        defects_by_beams = {}
        
        # Extract beam index from beam_key
        beam_idx = beam_key.split('_')[1]
        
        sequences_by_beams[beam_idx] = {}
        labels_by_beams[beam_idx] = {}
        defects_by_beams[beam_idx] = {}
        
        # Sort scan keys by index
        scans_keys_sorted = sorted(beam_data.keys(), key=lambda x: int(x.split('_')[0]))
        
        # Skip if not enough scans
        if len(scans_keys_sorted) < self.seq_length:
            return {}, {}, {}
        
        # Extract all scans, labels, and defects for this beam
        all_scans_for_beam = {}
        all_lbls_for_beam = {}
        all_defects_for_beam = {}
        
        # Use sequential scan index
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
            # Create a new sequence
            sequence = []
            labels = []
            defects = []
            
            # Determine start and end indices for this sequence
            if i < num_of_seqs_for_beam - 1:
                start_idx = i * self.seq_length
                end_idx = start_idx + self.seq_length
            else:
                # For the last sequence, ensure we have a full sequence length
                start_idx = len(scans_keys_sorted) - self.seq_length
                end_idx = len(scans_keys_sorted)
            
            # Skip if start_idx is negative
            if start_idx < 0:
                continue
            
            # Extract signals, labels, and defect positions for this sequence
            for j in range(start_idx, end_idx):
                scan_data = all_scans_for_beam[str(j)]
                labels.append(all_lbls_for_beam[str(j)])
                defects.append(all_defects_for_beam[str(j)])
                sequence.append(scan_data)
            
            # Skip if sequence doesn't have exactly seq_length signals
            if len(sequence) != self.seq_length:
                continue
            
            # Add to dataset
            sequences_by_beams[beam_idx][str(i)] = sequence
            labels_by_beams[beam_idx][str(i)] = labels
            defects_by_beams[beam_idx][str(i)] = defects
        
        return sequences_by_beams, labels_by_beams, defects_by_beams


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
        beam_layout = QHBoxLayout()
        self.beam_label = QLabel("Beam:")
        self.beam_combo = QComboBox()
        self.beam_combo.currentIndexChanged.connect(self.update_scan_combo)
        self.beam_slider = QSlider(Qt.Orientation.Horizontal)
        self.beam_slider.setMinimum(0)
        self.beam_slider.setMaximum(0)  # Will be updated when beams are loaded
        self.beam_slider.valueChanged.connect(self.on_beam_slider_changed)
        beam_layout.addWidget(self.beam_label)
        beam_layout.addWidget(self.beam_combo, 1)
        beam_layout.addWidget(self.beam_slider, 2)
        
        # Sequence selection
        sequence_layout = QHBoxLayout()
        self.sequence_label = QLabel("Sequence:")
        self.sequence_combo = QComboBox()
        self.sequence_combo.currentIndexChanged.connect(self.update_sequence_visualization)
        self.sequence_slider = QSlider(Qt.Orientation.Horizontal)
        self.sequence_slider.setMinimum(0)
        self.sequence_slider.setMaximum(0)  # Will be updated when sequences are loaded
        self.sequence_slider.valueChanged.connect(self.on_sequence_slider_changed)
        sequence_layout.addWidget(self.sequence_label)
        sequence_layout.addWidget(self.sequence_combo, 1)
        sequence_layout.addWidget(self.sequence_slider, 2)
        
        # Signal selection
        signal_layout = QHBoxLayout()
        self.signal_label = QLabel("Signal:")
        self.signal_combo = QComboBox()
        self.signal_combo.currentIndexChanged.connect(self.update_signal_visualization)
        self.signal_slider = QSlider(Qt.Orientation.Horizontal)
        self.signal_slider.setMinimum(0)
        self.signal_slider.setMaximum(0)  # Will be updated when signals are loaded
        self.signal_slider.valueChanged.connect(self.on_signal_slider_changed)
        signal_layout.addWidget(self.signal_label)
        signal_layout.addWidget(self.signal_combo, 1)
        signal_layout.addWidget(self.signal_slider, 2)
        
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
        
        # Add selection layouts
        beam_widget = QWidget()
        beam_widget.setLayout(beam_layout)
        controls_layout.addWidget(beam_widget, 3, 0, 1, 4)
        
        sequence_widget = QWidget()
        sequence_widget.setLayout(sequence_layout)
        controls_layout.addWidget(sequence_widget, 4, 0, 1, 4)
        
        signal_widget = QWidget()
        signal_widget.setLayout(signal_layout)
        controls_layout.addWidget(signal_widget, 5, 0, 1, 4)
        
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
            
            # Update beam slider
            self.beam_slider.setMaximum(len(self.beam_keys) - 1)
            self.beam_slider.setValue(0)
            
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
            
            # Update sequence slider
            self.sequence_slider.setMaximum(len(self.sequence_indices) - 1 if self.sequence_indices else 0)
            self.sequence_slider.setValue(0)
        else:
            self.sequence_combo.clear()
            self.sequence_indices = []
            self.sequence_slider.setMaximum(0)
            self.sequence_slider.setValue(0)
    
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
            
            # Update signal slider
            self.signal_slider.setMaximum(len(signal_keys) - 1 if signal_keys else 0)
            self.signal_slider.setValue(0)
            
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
        
        # Transpose the array to swap axes (signal position on Y, signal index on X)
        signals_array = signals_array.T
        
        # Display as an image with origin='lower' to flip the Y-axis (0 at bottom)
        im = ax.imshow(signals_array, aspect='auto', cmap='viridis', origin='lower')
        
        # Add colorbar
        self.sequence_figure.colorbar(im, ax=ax, label='Signal Value')
        
        # Mark ground truth defects
        for i, (label, position) in enumerate(zip(labels, gt_positions)):
            if label > 0.5:  # If it's a defect
                start, end = position
                start_idx = int(start * signals_array.shape[0])
                end_idx = int(end * signals_array.shape[0])
                rect = plt.Rectangle((i - 0.5, start_idx), 1, end_idx - start_idx, 
                                    fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
        
        # Mark predictions
        for i, (prob, position) in enumerate(zip(pred_probs, pred_positions)):
            if prob > 0.5:  # If prediction confidence is high enough
                start, end = position
                start_idx = int(start * signals_array.shape[0])
                end_idx = int(end * signals_array.shape[0])
                rect = plt.Rectangle((i - 0.5, start_idx), 1, end_idx - start_idx, 
                                    fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(rect)
                ax.text(i, start_idx, f"{prob:.2f}", color='blue', fontsize=8)
        
        # Add labels and title
        ax.set_xlabel('Scan Index')
        ax.set_ylabel('Depth (in signal indexes)')
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
    
    def on_beam_slider_changed(self, value):
        """Handle beam slider value change"""
        if 0 <= value < self.beam_combo.count():
            self.beam_combo.setCurrentIndex(value)
    
    def on_sequence_slider_changed(self, value):
        """Handle sequence slider value change"""
        if 0 <= value < self.sequence_combo.count():
            self.sequence_combo.setCurrentIndex(value)
    
    def on_signal_slider_changed(self, value):
        """Handle signal slider value change"""
        if 0 <= value < self.signal_combo.count():
            self.signal_combo.setCurrentIndex(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelTesterApp()
    window.show()
    sys.exit(app.exec())
