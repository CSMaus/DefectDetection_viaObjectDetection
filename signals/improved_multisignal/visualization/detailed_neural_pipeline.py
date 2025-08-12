#!/usr/bin/env python3
"""
DETAILED Neural Network Pipeline Visualization
Shows REAL HybridBinaryModel processing with REAL data in extreme detail:

1. RealSignalSequenceVisualization - Show REAL 50-signal sequence with actual defect locations
2. RealFeatureExtractionVisualization - Show REAL conv layers, pooling, shared layer outputs  
3. RealTransformerInputVisualization - Show REAL 64-dim feature vectors with actual values
4. RealTransformerProcessingVisualization - Show REAL attention weights and layer transformations
5. RealBinaryClassificationVisualization - Show REAL predictions for each signal with confidence

All using YOUR REAL HybridBinaryModel and YOUR REAL training data!
"""

from manim import *
import numpy as np
import sys
import os
import torch
import json

# Import real model and data (script runs from visualization/, so go up to improved_multisignal/)
sys.path.append('..')
from detection_models.hybrid_binary import HybridBinaryModel
from json_dataset import JsonSignalDataset
from real_data_loader import RealPAUTDataLoader

# Import debug control
try:
    from run_visualizations import DEBUG_PRINTS
except ImportError:
    DEBUG_PRINTS = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINTS is True"""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

class NeuralNetworkDataLoader:
    """Load real data and model for neural network visualization"""
    
    def __init__(self):
        # Script runs from visualization/ folder, need to go up to improved_multisignal/
        self.model_path = "../models/HybridBinaryModel_20250718_2100/best_detection.pth"
        self.data_path = "../json_data"
        self.model = None
        self.real_sequence = None
        self.real_labels = None
        self.model_features = None
        
    def load_model_and_data(self):
        """Load the trained model and real data"""
        debug_print("Loading HybridBinaryModel and real data...")
        
        # Load model
        try:
            self.model = HybridBinaryModel()
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Full training checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                debug_print("Model loaded from checkpoint with model_state_dict")
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
                debug_print("Model loaded from direct state dict")
                
            self.model.eval()
        except Exception as e:
            debug_print(f"Error loading model: {e}")
            return False

        # Load REAL data using your existing method
        try:
            from real_data_loader import get_real_data_sample
            loader, filename = get_real_data_sample()
            
            if not loader or not filename:
                debug_print("No real data files found")
                return False
            
            # Get defect sequences using your existing method
            sequences = loader.get_defect_sequences(filename, seq_length=50)
            if not sequences:
                debug_print(f"No sequences found in {filename}")
                return False
            
            # Find sequence with most defects
            best_sequence = None
            max_defects = 0
            
            for sequence in sequences:
                defect_count = sum(sequence['labels'])
                if defect_count > max_defects:
                    max_defects = defect_count
                    best_sequence = sequence
            
            if best_sequence:
                self.real_sequence = np.array(best_sequence['signals'])
                self.real_labels = np.array(best_sequence['labels'])
                self.defect_info = best_sequence['defect_info']
                self.depth_range = best_sequence['depth_range']
                debug_print(f"Using REAL sequence with {max_defects} defects from {filename}")
            else:
                # Use first sequence
                self.real_sequence = np.array(sequences[0]['signals'])
                self.real_labels = np.array(sequences[0]['labels'])
                self.defect_info = sequences[0]['defect_info']
                self.depth_range = sequences[0]['depth_range']
                debug_print(f"Using first sequence from {filename}")
                
        except Exception as e:
            debug_print(f"Error loading real data: {e}")
            return False
        
        # Extract model features
        self._extract_model_features()
        return True
    
    def _extract_model_features(self):
        """Extract intermediate features from the model"""
        debug_print("Extracting model features...")
        
        with torch.no_grad():
            x = torch.tensor(self.real_sequence).unsqueeze(0).float()
            batch_size, num_signals, signal_length = x.size()
            
            # FOLLOW EXACT HybridBinaryModel ARCHITECTURE
            # 1. Reshape for conv layers
            x_conv = x.view(batch_size * num_signals, 1, signal_length)
            
            # 2. Conv layers processing
            conv_features = self.model.conv_layers(x_conv)
            
            # 3. Fixed pooling
            pooled_features = self.model.fixed_pool(conv_features)
            pooled_features = torch.nn.functional.interpolate(pooled_features, size=128, mode='linear', align_corners=False)
            pooled_features = pooled_features.mean(dim=1)  # Global average pooling
            
            # 4. Shared layer features
            shared_features = self.model.shared_layer(pooled_features)
            shared_features = shared_features.view(batch_size, num_signals, -1)
            
            # 5. Positional encoding
            pos_encoded = self.model.position_encoding(shared_features)
            
            # 6. Transformer features
            transformer_features = pos_encoded
            for transformer in self.model.transformer_layers:
                transformer_features = transformer(transformer_features)
            
            # 7. Final predictions
            predictions = self.model.classifier(transformer_features).squeeze(-1)
            predictions = torch.sigmoid(predictions)
            
            self.model_features = {
                'conv_features': conv_features.view(batch_size, num_signals, -1).numpy()[0],
                'shared_features': shared_features.numpy()[0],
                'pos_encoded': pos_encoded.numpy()[0],
                'transformer_features': transformer_features.numpy()[0],
                'predictions': predictions.numpy()[0]
            }
            
        debug_print("Model features extracted")

# Global data loader instance
data_loader = NeuralNetworkDataLoader()

class SignalInputVisualization(Scene):
    """Video 1: Show signal input representation - ONLY THIS ONE"""
    
    def construct(self):
        # Load data
        if not data_loader.load_model_and_data():
            debug_print("Failed to load data for SignalInputVisualization")
            return
            
        # NO TITLE - removed completely
        
        # Phase 1: Show first 4 signals individually
        self.show_individual_signals()
        
        # Phase 2: Show dots and last 2 signals
        self.show_signals_with_dots()
        
        # Phase 3: Move signals to side and show parallel feature extraction
        self.show_parallel_feature_extraction()
    
    def show_individual_signals(self):
        """Show first 4 signals as individual plots"""
        debug_print("Phase 1: Individual signals")
        
        signal_plots = VGroup()
        
        for i in range(4):
            # Get signal data
            signal_data = data_loader.real_sequence[i]
            is_defect = data_loader.real_labels[i] > 0.5
            info = data_loader.defect_info[i]
            prediction = data_loader.model_features['predictions'][i]
            
            # Create signal plot
            axes = Axes(
                x_range=[0, len(signal_data), len(signal_data)//4],
                y_range=[np.min(signal_data)-0.1, np.max(signal_data)+0.1, 0.2],
                x_length=6,
                y_length=1.5,
                axis_config={"color": WHITE, "stroke_width": 1, "tip_length": 0.02}
            )
            axes.shift(UP * (2 - i * 1.2))
            
            # Plot signal
            signal_curve = axes.plot(
                lambda x: np.interp(x, np.arange(len(signal_data)), signal_data),
                color=RED if is_defect else BLUE,
                stroke_width=2
            )
            
            # Add label
            label = Text(f"Signal {i+1}", font_size=16, color=WHITE)
            label.next_to(axes, LEFT)
            
            # Add defect info if present
            if is_defect and info.get('start', 0) > 0:
                # Highlight defect region
                start_sample = info['start_sample']
                end_sample = info['end_sample']
                
                if start_sample < len(signal_data) and end_sample <= len(signal_data):
                    highlight_width = (end_sample - start_sample) / len(signal_data) * 6
                    highlight_center_x = axes.c2p((start_sample + end_sample) / 2, 0)[0]
                    
                    defect_highlight = Rectangle(
                        width=highlight_width, height=1.7,
                        color=YELLOW, fill_opacity=0.3, stroke_width=2
                    )
                    defect_highlight.move_to([highlight_center_x, axes.get_center()[1], 0])
                    
                    defect_text = Text(
                        f"DEFECT: {info['start']:.1f}-{info['end']:.1f}mm",
                        font_size=12, color=RED
                    )
                    defect_text.next_to(axes, RIGHT)
                    
                    signal_plots.add(axes, signal_curve, label, defect_highlight, defect_text)
                else:
                    defect_text = Text(f"DEFECT", font_size=12, color=RED)
                    defect_text.next_to(axes, RIGHT)
                    signal_plots.add(axes, signal_curve, label, defect_text)
            else:
                normal_text = Text(f"NORMAL", font_size=12, color=BLUE)
                normal_text.next_to(axes, RIGHT)
                signal_plots.add(axes, signal_curve, label, normal_text)
        
        self.play(Create(signal_plots), run_time=3)
        self.wait(1)
        
        # Store for next phase
        self.signal_plots = signal_plots
    
    def show_signals_with_dots(self):
        """Show dots indicating more signals and last 2 signals"""
        debug_print("Phase 2: Dots and last signals")
        
        # Shrink existing signals
        self.play(self.signal_plots.animate.scale(0.6).shift(UP * 1.5), run_time=1)
        
        # Add dots
        dots = VGroup()
        for i in range(3):
            dot = Dot(color=YELLOW, radius=0.1)
            dot.shift(DOWN * (0.2 + i * 0.2))
            dots.add(dot)
        
        # Count middle signals
        middle_defects = sum(data_loader.real_labels[4:48])
        middle_normal = 44 - middle_defects
        
        dots_text = Text(
            f"... 44 more signals ...\n{middle_defects} defects, {middle_normal} normal",
            font_size=14, color=YELLOW
        )
        dots_text.next_to(dots, DOWN)
        
        self.play(Create(dots), Write(dots_text), run_time=1)
        
        # Add last 2 signals
        last_signals = VGroup()
        for i, signal_idx in enumerate([48, 49]):
            signal_data = data_loader.real_sequence[signal_idx]
            is_defect = data_loader.real_labels[signal_idx] > 0.5
            info = data_loader.defect_info[signal_idx]
            
            axes = Axes(
                x_range=[0, len(signal_data), len(signal_data)//4],
                y_range=[np.min(signal_data)-0.1, np.max(signal_data)+0.1, 0.2],
                x_length=3.5,
                y_length=0.8,
                axis_config={"color": WHITE, "stroke_width": 1, "tip_length": 0.02}
            )
            axes.shift(DOWN * (2.2 + i * 1.0))
            
            signal_curve = axes.plot(
                lambda x: np.interp(x, np.arange(len(signal_data)), signal_data),
                color=RED if is_defect else BLUE,
                stroke_width=2
            )
            
            label = Text(f"Signal {signal_idx+1}", font_size=12, color=WHITE)
            label.next_to(axes, LEFT)
            
            if is_defect and info.get('start', 0) > 0:
                status_text = f"DEFECT: {info['start']:.1f}-{info['end']:.1f}mm"
            else:
                status_text = f"{'DEFECT' if is_defect else 'NORMAL'}"
            
            status_label = Text(status_text, font_size=8, color=RED if is_defect else BLUE)
            status_label.next_to(axes, RIGHT)
            
            last_signals.add(axes, signal_curve, label, status_label)
        
        self.play(Create(last_signals), run_time=2)
        self.wait(1)
        
        # Store for next phase
        self.dots = dots
        self.dots_text = dots_text
        self.last_signals = last_signals
    
    def show_parallel_feature_extraction(self):
        # Move all signals to the left and make them BIGGER
        all_signals = VGroup(self.signal_plots, self.dots, self.dots_text, self.last_signals)
        self.play(all_signals.animate.scale(0.6).shift(LEFT * 4.5), run_time=2)

        # Layout - adjusted for 6 arrows and better spacing
        lanes_to_show = 6  # 6 arrows as requested
        lane_gap = 0.6
        top_y = 1.8
        x_in = -1.0  # lane input x (to the right of moved signals)
        x_cnn = 1.2  # per-lane CNN block x
        x_out = 4.8  # feature output x

        # Shared weights box (θ) above lanes
        theta_box = Rectangle(width=2.2, height=0.6, color=YELLOW, fill_opacity=0.15)
        theta_box.move_to([x_cnn, top_y + 1.1, 0])
        theta_text = Text("Shared weights θ", font_size=22)
        theta_text.move_to(theta_box.get_center())

        # Per-lane identical CNN blocks + arrows + outputs
        lane_groups = VGroup()
        weight_lines = VGroup()
        out_arrows = VGroup()
        out_feats = VGroup()

        # Use real shared feature stats to vary opacity slightly (purely visual)
        shared_feats = data_loader.model_features.get('shared_features', None)
        for i in range(lanes_to_show):
            y = top_y - i * lane_gap

            # Per-lane CNN block (identical block = same architecture)
            cnn_block = Rectangle(width=1.6, height=0.45, color=GREEN, fill_opacity=0.85)
            cnn_block.move_to([x_cnn, y, 0])
            cnn_label = Text("CNN", font_size=18, color=WHITE).move_to(cnn_block.get_center())

            # STRAIGHT HORIZONTAL arrows (no lines in front, no angles)
            to_cnn = Arrow(start=[x_in, y, 0], end=[x_cnn - 0.9, y, 0],
                           color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)
            to_out = Arrow(start=[x_cnn + 0.9, y, 0], end=[x_out - 0.6, y, 0],
                           color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.1)

            # Output feature rectangle (64-dim per signal)
            feat = Rectangle(width=0.7, height=0.35, color=ORANGE, fill_opacity=0.8)
            feat.move_to([x_out, y, 0])
            feat_text = Text("64", font_size=16).move_to(feat.get_center())

            # Slight opacity cue from real data (no coupling between lanes)
            if shared_feats is not None and i < len(shared_feats):
                val = float(np.mean(shared_feats[i]))
                feat.set_fill(ORANGE, opacity=0.35 + 0.45 * min(1.0, abs(val) / 2.0))

            # Dashed “broadcast” line from shared θ → per-lane CNN (shows weight sharing)
            wline = DashedLine(theta_box.get_bottom(), cnn_block.get_top(), dash_length=0.12, color=YELLOW)

            lane_groups.add(VGroup(to_cnn, cnn_block, cnn_label))
            weight_lines.add(wline)
            out_arrows.add(to_out)
            out_feats.add(VGroup(feat, feat_text))

        # Labels
        lanes_label = Text("×50 lanes in parallel (no mixing)", font_size=22)
        lanes_label.next_to(out_feats, DOWN, buff=0.6)
        out_label = Text("Per-signal feature vectors (64-dim each)", font_size=22)
        out_label.next_to(out_feats, RIGHT, buff=0.4)

        # Animate
        self.play(Create(theta_box), Write(theta_text), run_time=0.6)
        self.play(LaggedStart(*[Create(g) for g in lane_groups], lag_ratio=0.05), run_time=1.2)
        self.play(LaggedStart(*[Create(l) for l in weight_lines], lag_ratio=0.05), run_time=0.8)
        self.play(LaggedStart(*[Create(a) for a in out_arrows], lag_ratio=0.05), run_time=0.6)
        self.play(LaggedStart(*[Create(f) for f in out_feats], lag_ratio=0.05), run_time=0.8)
        self.play(Write(lanes_label), Write(out_label), run_time=0.6)

        # Subtle pulse to emphasize simultaneous processing
        for _ in range(2):
            self.play(theta_box.animate.set_fill(opacity=0.25), run_time=0.25)
            self.play(theta_box.animate.set_fill(opacity=0.15), run_time=0.25)

        self.wait(3)
