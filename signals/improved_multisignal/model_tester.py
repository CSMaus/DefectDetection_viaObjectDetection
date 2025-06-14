import os
import sys
import json
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QComboBox, QPushButton, QLabel, QFileDialog, QFrame, QGridLayout,
                            QSpinBox, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from improved_model import ImprovedMultiSignalClassifier


class PredictionWorker(QThread):
    """Worker thread for running predictions"""
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
            
            # Process each scan key in this beam
            beam_results = {}
            for scan_key, scan_data in beam_data.items():
                # Extract signals
                signal_keys = sorted(scan_data.keys(), 
                                    key=lambda x: int(round(float(x.split('_')[0]))))
                
                # Process in sequences of seq_length
                for start_idx in range(0, len(signal_keys), self.seq_length):
                    end_idx = min(start_idx + self.seq_length, len(signal_keys))
                    seq_keys = signal_keys[start_idx:end_idx]
                    
                    signals = []
                    labels = []
                    defect_positions = []
                    
                    for signal_key in seq_keys:
                        signal_info = scan_data[signal_key]
                        signal = np.array(signal_info['signal'], dtype=np.float32)
                        signals.append(signal)
                        
                        defect_name = signal_key.split('_')[1]
                        if defect_name == 'Health':
                            labels.append(0.0)
                            defect_positions.append([0.0, 0.0])
                        else:
                            # Extract defect position from key
                            defect_range = signal_key.split('_')[2].split('-')
                            defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                            labels.append(1.0)
                            defect_positions.append([defect_start, defect_end])
                    
                    # Pad if needed
                    if len(signals) < self.seq_length:
                        pad_length = self.seq_length - len(signals)
                        signal_length = len(signals[0])
                        
                        signals.extend([np.zeros(signal_length, dtype=np.float32) for _ in range(pad_length)])
                        labels.extend([0.0 for _ in range(pad_length)])
                        defect_positions.extend([[0.0, 0.0] for _ in range(pad_length)])
                    
                    # Convert to tensors
                    signals_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Run prediction
                    with torch.no_grad():
                        defect_prob, defect_start, defect_end = self.model(signals_tensor)
                    
                    # Store results
                    for i, signal_key in enumerate(seq_keys):
                        if i < len(defect_prob[0]):
                            beam_results[signal_key] = {
                                'gt_label': labels[i],
                                'gt_position': defect_positions[i],
                                'pred_prob': defect_prob[0][i].item(),
                                'pred_start': defect_start[0][i].item(),
                                'pred_end': defect_end[0][i].item(),
                                'signal': signals[i].tolist()
                            }
            
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
        
        # Scan selection
        self.scan_label = QLabel("Scan:")
        self.scan_combo = QComboBox()
        self.scan_combo.currentIndexChanged.connect(self.update_signal_combo)
        
        # Signal selection
        self.signal_label = QLabel("Signal:")
        self.signal_combo = QComboBox()
        self.signal_combo.currentIndexChanged.connect(self.update_visualization)
        
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
        controls_layout.addWidget(self.scan_label, 3, 2)
        controls_layout.addWidget(self.scan_combo, 3, 3)
        controls_layout.addWidget(self.signal_label, 4, 0)
        controls_layout.addWidget(self.signal_combo, 4, 1, 1, 3)
        
        # Visualization area
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add widgets to main layout
        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.canvas)
        
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
        """Update scan combo box based on selected beam"""
        if index < 0 or not self.beam_keys:
            return
        
        beam_key = self.beam_keys[index]
        
        if beam_key in self.prediction_results:
            # Get unique scan keys from the prediction results
            signal_keys = list(self.prediction_results[beam_key].keys())
            scan_keys = set()
            for key in signal_keys:
                scan_key = key.split('_')[0]
                scan_keys.add(scan_key)
            
            self.scan_keys = sorted(list(scan_keys))
            self.scan_combo.clear()
            self.scan_combo.addItems(self.scan_keys)
        else:
            self.scan_combo.clear()
            self.scan_keys = []
    
    def update_signal_combo(self, index):
        """Update signal combo box based on selected scan"""
        if index < 0 or not self.scan_keys or self.beam_combo.currentIndex() < 0:
            return
        
        beam_key = self.beam_keys[self.beam_combo.currentIndex()]
        scan_key = self.scan_keys[index]
        
        if beam_key in self.prediction_results:
            # Get signal keys for this beam and scan
            signal_keys = []
            for key in self.prediction_results[beam_key].keys():
                if key.startswith(f"{scan_key}_"):
                    signal_keys.append(key)
            
            self.signal_keys = sorted(signal_keys, 
                                     key=lambda x: int(round(float(x.split('_')[0]))))
            self.signal_combo.clear()
            self.signal_combo.addItems(self.signal_keys)
        else:
            self.signal_combo.clear()
            self.signal_keys = []
    
    def update_visualization(self, index):
        """Update visualization for selected signal"""
        if index < 0 or not self.signal_keys or self.beam_combo.currentIndex() < 0 or self.scan_combo.currentIndex() < 0:
            return
        
        beam_key = self.beam_keys[self.beam_combo.currentIndex()]
        signal_key = self.signal_keys[index]
        
        if beam_key in self.prediction_results and signal_key in self.prediction_results[beam_key]:
            result = self.prediction_results[beam_key][signal_key]
            
            # Clear figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
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
            ax.set_title(f"Signal: {signal_key}\nPrediction: {result['pred_prob']:.4f}")
            if (result['gt_label'] > 0.5) or (result['pred_prob'] > 0.5):
                ax.legend()
            
            # Add grid
            ax.grid(True)
            
            # Update canvas
            self.canvas.draw()
    
    def show_error(self, message):
        """Show error message"""
        self.predict_status.setText(f"Error: {message}")
        print(f"Error: {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelTesterApp()
    window.show()
    sys.exit(app.exec())
