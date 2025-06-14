import os
import sys
import json
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QComboBox, QPushButton, QLabel,
                             QFileDialog, QSplitter, QFrame, QSlider, QGridLayout)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeyEvent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dataset_preparation import SignalSequencePreparation
from two_stage_model import TwoStageDefectDetector


class SignalVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Defect Detector")
        self.setGeometry(100, 100, 1200, 800)

        # Model and data attributes
        self.model = None
        self.model_path = None
        self.json_folder = None
        self.json_files = []
        self.current_json_file = None
        self.sequences = {}
        self.annotations = {}
        self.current_sequence_idx = 0
        self.current_signal_idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup UI
        self.setup_ui()
        
        # Install event filter for key press events
        self.installEventFilter(self)

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Controls area
        controls_frame = QFrame()
        controls_layout = QGridLayout(controls_frame)
        
        # JSON folder selection
        self.folder_label = QLabel("JSON Folder:")
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_json_folder)
        
        # JSON file selection
        self.json_label = QLabel("JSON File:")
        self.json_combo = QComboBox()
        self.json_combo.currentIndexChanged.connect(self.load_json_file)
        
        # Model selection
        self.model_label = QLabel("Model:")
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.load_model)
        self.model_path_label = QLabel("No model loaded")
        
        # Sequence selection with slider
        self.sequence_label = QLabel("Sequence:")
        self.sequence_slider = QSlider(Qt.Orientation.Horizontal)
        self.sequence_slider.setMinimum(0)
        self.sequence_slider.setMaximum(0)
        self.sequence_slider.valueChanged.connect(self.update_sequence)
        self.sequence_name_label = QLabel("No sequence selected")
        
        # Signal selection with slider
        self.signal_label = QLabel("Signal:")
        self.signal_slider = QSlider(Qt.Orientation.Horizontal)
        self.signal_slider.setMinimum(0)
        self.signal_slider.setMaximum(0)
        self.signal_slider.valueChanged.connect(self.update_visualization)
        self.signal_idx_label = QLabel("No signal selected")
        
        # Navigation buttons for sequences
        self.prev_seq_button = QPushButton("Previous Sequence")
        self.prev_seq_button.clicked.connect(self.previous_sequence)
        self.next_seq_button = QPushButton("Next Sequence")
        self.next_seq_button.clicked.connect(self.next_sequence)
        
        # Navigation buttons for signals
        self.prev_signal_button = QPushButton("Previous Signal")
        self.prev_signal_button.clicked.connect(self.previous_signal)
        self.next_signal_button = QPushButton("Next Signal")
        self.next_signal_button.clicked.connect(self.next_signal)
        
        # Add controls to layout
        controls_layout.addWidget(self.folder_label, 0, 0)
        controls_layout.addWidget(self.folder_button, 0, 1)
        controls_layout.addWidget(self.json_label, 0, 2)
        controls_layout.addWidget(self.json_combo, 0, 3)
        controls_layout.addWidget(self.model_label, 1, 0)
        controls_layout.addWidget(self.model_button, 1, 1)
        controls_layout.addWidget(self.model_path_label, 1, 2, 1, 2)
        controls_layout.addWidget(self.sequence_label, 2, 0)
        controls_layout.addWidget(self.sequence_slider, 2, 1, 1, 2)
        controls_layout.addWidget(self.sequence_name_label, 2, 3)
        controls_layout.addWidget(self.prev_seq_button, 3, 0)
        controls_layout.addWidget(self.next_seq_button, 3, 1)
        controls_layout.addWidget(self.signal_label, 4, 0)
        controls_layout.addWidget(self.signal_slider, 4, 1, 1, 2)
        controls_layout.addWidget(self.signal_idx_label, 4, 3)
        controls_layout.addWidget(self.prev_signal_button, 5, 0)
        controls_layout.addWidget(self.next_signal_button, 5, 1)
        
        # Visualization area
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add widgets to main layout
        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.canvas)
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def eventFilter(self, obj, event):
        """Handle key press events for navigation"""
        if event.type() == QEvent.Type.KeyPress:
            key_event = QKeyEvent(event)
            if key_event.key() == Qt.Key.Key_Left:
                self.previous_signal()
                return True
            elif key_event.key() == Qt.Key.Key_Right:
                self.next_signal()
                return True
            elif key_event.key() == Qt.Key.Key_Up:
                self.previous_sequence()
                return True
            elif key_event.key() == Qt.Key.Key_Down:
                self.next_sequence()
                return True
        return super().eventFilter(obj, event)
    
    def load_model(self):
        """Load the trained model"""
        # Ask user to select model directory
        model_dir = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if not model_dir:
            return
        
        # Find best model file
        model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            self.show_error("Model file not found")
            return
        
        # Store model path
        self.model_path = model_path
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model instance (assuming signal length of 320, can be adjusted)
        signal_length = 320  # This will be updated when we load actual data
        self.model = TwoStageDefectDetector(
            signal_length=signal_length,
            d_model=128,
            num_classes=2  # Assuming binary classification
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Update model path label
        self.model_path_label.setText(os.path.basename(model_dir))
        
        print(f"Model loaded from {model_path}")
        
        # Update visualization if we have data
        if self.sequences:
            self.update_visualization(self.signal_slider.value())
    
    def select_json_folder(self):
        """Select folder containing JSON files"""
        folder = QFileDialog.getExistingDirectory(self, "Select JSON Folder")
        if not folder:
            return
        
        self.json_folder = folder
        self.json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        
        # Update JSON combo box
        self.json_combo.clear()
        self.json_combo.addItems(self.json_files)
    
    def load_json_file(self, index):
        """Load selected JSON file and extract sequences"""
        if index < 0 or not self.json_folder:
            return
        
        json_file = self.json_files[index]
        self.current_json_file = json_file
        json_path = os.path.join(self.json_folder, json_file)
        
        # Create a temporary SignalSequencePreparation instance
        prep = SignalSequencePreparation(self.json_folder, "temp_output", seq_length=50)
        
        # Create the full sequences with padding to 50 signals
        prep.create_signal_sequences()
        
        # Get the beam sequences for this file
        beam_sequences = []
        for seq in prep.create_beam_sequences():
            if seq['file_name'] == os.path.splitext(json_file)[0]:
                beam_sequences.append(seq)
        
        if not beam_sequences:
            self.show_error(f"No sequences found for {json_file}")
            return
        
        # Convert to the format expected by the visualizer
        sequences = {}
        annotations = {}
        
        for i, seq in enumerate(beam_sequences):
            scan_key = f"{seq['scan_key']}_{i}"
            sequences[scan_key] = seq['signals']
            
            # Get annotations for this sequence
            if 'annotations' in seq:
                annotations[scan_key] = seq['annotations']
        
        self.sequences = sequences
        self.annotations = annotations
        
        # Update sequence slider
        sequence_keys = list(sequences.keys())
        self.sequence_slider.setMaximum(len(sequence_keys) - 1)
        self.sequence_slider.setValue(0)
        self.current_sequence_idx = 0
        
        # Update sequence name label
        if sequence_keys:
            self.sequence_name_label.setText(f"{json_file} - Sequence: {sequence_keys[0]}")
            
            # Update signal slider
            scan_key = sequence_keys[0]
            if scan_key in sequences:
                signals = sequences[scan_key]
                self.signal_slider.setMaximum(len(signals) - 1)
                self.signal_slider.setValue(0)
                self.current_signal_idx = 0
                self.signal_idx_label.setText(f"Signal: 0/{len(signals) - 1}")
                
                # Print debug info
                print(f"Loaded sequence {scan_key} with {len(signals)} signals")
        
        # Update model's signal length based on actual data
        if self.model and sequences and sequence_keys:
            first_key = sequence_keys[0]
            if first_key in sequences and len(sequences[first_key]) > 0:
                signal_length = sequences[first_key][0].shape[0]  # Length of first signal
                
                # Recreate model with correct signal length
                self.model = TwoStageDefectDetector(
                    signal_length=signal_length,
                    d_model=128,
                    num_classes=2
                ).to(self.device)
                
                # Reload weights if we have a model path
                if self.model_path and os.path.exists(self.model_path):
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
        
        # Update visualization
        self.update_visualization(0)
    
    def update_sequence(self, index):
        """Update for selected sequence"""
        if index < 0 or not self.sequences:
            return
        
        self.current_sequence_idx = index
        sequence_keys = list(self.sequences.keys())
        if index >= len(sequence_keys):
            return
        
        scan_key = sequence_keys[index]
        signals = self.sequences[scan_key]
        
        # Update sequence name label
        if self.current_json_file:
            self.sequence_name_label.setText(f"{self.current_json_file} - Sequence: {scan_key}")
        
        # Update signal slider
        self.signal_slider.setMaximum(len(signals) - 1)
        self.signal_slider.setValue(0)
        self.current_signal_idx = 0
        self.signal_idx_label.setText(f"Signal: 0/{len(signals) - 1}")
        
        # Update visualization
        self.update_visualization(0)
    
    def previous_sequence(self):
        """Go to previous sequence"""
        if not self.sequences:
            return
        
        new_idx = max(0, self.sequence_slider.value() - 1)
        self.sequence_slider.setValue(new_idx)
    
    def next_sequence(self):
        """Go to next sequence"""
        if not self.sequences:
            return
        
        new_idx = min(self.sequence_slider.maximum(), self.sequence_slider.value() + 1)
        self.sequence_slider.setValue(new_idx)
    
    def previous_signal(self):
        """Go to previous signal"""
        if not self.sequences:
            return
        
        new_idx = max(0, self.signal_slider.value() - 1)
        self.signal_slider.setValue(new_idx)
    
    def next_signal(self):
        """Go to next signal"""
        if not self.sequences:
            return
        
        new_idx = min(self.signal_slider.maximum(), self.signal_slider.value() + 1)
        self.signal_slider.setValue(new_idx)
    
    def update_visualization(self, index):
        """Update visualization for selected signal"""
        if index < 0 or not self.sequences:
            return
        
        self.current_signal_idx = index
        sequence_keys = list(self.sequences.keys())
        if self.current_sequence_idx >= len(sequence_keys):
            return
        
        scan_key = sequence_keys[self.current_sequence_idx]
        signals = self.sequences[scan_key]
        
        if index >= len(signals):
            return
        
        # Update signal index label
        self.signal_idx_label.setText(f"Signal: {index}/{len(signals) - 1}")
        
        # Get the current signal
        signal = signals[index]
        
        # Clear figure
        self.figure.clear()
        
        # Plot the signal
        ax = self.figure.add_subplot(111)
        ax.set_title(f"Signal {index} from Sequence {scan_key}")
        ax.plot(signal, 'b-', label='Signal')
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # Check if there are ground truth annotations for this signal
        if scan_key in self.annotations:
            for defect in self.annotations[scan_key]:
                beam_start, beam_end = defect["bbox"][0], defect["bbox"][1]
                defect_start, defect_end = defect["bbox"][2], defect["bbox"][3]
                
                # Check if this signal is within the beam range
                beam_position = index / len(signals)
                if beam_start <= beam_position <= beam_end:
                    # Convert defect positions to signal indices
                    start_idx = int(defect_start * len(signal))
                    end_idx = int(defect_end * len(signal))
                    
                    # Highlight the defect region
                    ax.axvspan(start_idx, end_idx, alpha=0.3, color='red', label=f'GT: {defect["label"]}')
                    ax.text(start_idx, max(signal), f"GT: {defect['label']}", color='red', fontsize=10)
        
        # Make predictions with model if available
        if self.model:
            # Prepare input for model (single signal)
            input_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model.predict(input_tensor)
            
            # Check if there are predictions for this signal
            if predictions and len(predictions) > 0:
                for pred in predictions[0]:  # First batch
                    position = pred['position']  # This is the signal index
                    
                    # Check if this prediction is for the current signal
                    if position == index:
                        defect_pos = pred['defect_position']  # This is [start, end] within signal
                        confidence = pred['adjusted_confidence']
                        
                        # Convert defect positions to signal indices
                        start_idx = int(defect_pos[0] * len(signal))
                        end_idx = int(defect_pos[1] * len(signal))
                        
                        # Highlight the predicted defect region
                        ax.axvspan(start_idx, end_idx, alpha=0.3, color='blue', label=f'Pred: {confidence:.2f}')
                        ax.text(start_idx, min(signal), f"Pred: {confidence:.2f}", color='blue', fontsize=10)
        
        # Add legend
        ax.legend()
        
        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_error(self, message):
        """Show error message"""
        print(f"Error: {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalVisualizerApp()
    window.show()
    sys.exit(app.exec())
