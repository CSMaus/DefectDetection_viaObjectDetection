from manim import *
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from real_data_loader import get_real_data_sample

# Import DEBUG_PRINTS from run_visualizations
try:
    from run_visualizations import DEBUG_PRINTS
except ImportError:
    DEBUG_PRINTS = True  # Default to True if run_visualizations not available

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINTS is True"""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

class RealSignalProcessing(Scene):
    def construct(self):
        # Load real PAUT data
        loader, filename = get_real_data_sample()
        if not loader:
            print("Error: Could not load real PAUT data")
            return
        
        # Title (no animation)
        title = Text("Real PAUT Signal Analysis", font_size=32)
        title.to_edge(UP)
        self.add(title)
        
        # Get real clean vs defect signal comparison
        clean_signal, defect_signal, defect_info = loader.get_signal_comparison()
        if clean_signal is None or defect_signal is None:
            print("Could not find clean and defect signals")
            return
        
        # Show clean vs defect signal
        self.show_signal_comparison(clean_signal, defect_signal, defect_info)
        self.wait(2)
        
        # Show CNN feature extraction
        self.show_cnn_processing(defect_signal)
        self.wait(2)
        
        # Show transformer attention with real 50-signal sequence
        self.show_transformer_attention(loader)
        self.wait(2)
    
    def show_signal_comparison(self, clean_signal, defect_signal, defect_info):
        # Clean signal
        clean_axes = Axes(
            x_range=[0, len(clean_signal), len(clean_signal)//4],
            y_range=[np.min(clean_signal)-0.1, np.max(clean_signal)+0.1, 0.2],
            x_length=5,
            y_length=2,
            axis_config={"color": BLUE}
        ).move_to([-2, 1.5, 0])
        
        clean_label = Text("Clean Signal", font_size=20, color=BLUE)
        clean_label.next_to(clean_axes, UP)
        
        # Plot clean signal
        clean_points = []
        for i, val in enumerate(clean_signal[::4]):
            clean_points.append(clean_axes.c2p(i*4, val))
        
        clean_curve = VMobject(color=BLUE)
        clean_curve.set_points_smoothly(clean_points)
        
        # Defect signal
        defect_axes = Axes(
            x_range=[0, len(defect_signal), len(defect_signal)//4],
            y_range=[np.min(defect_signal)-0.1, np.max(defect_signal)+0.1, 0.2],
            x_length=5,
            y_length=2,
            axis_config={"color": RED}
        ).move_to([2, 1.5, 0])
        
        defect_label = Text("Defect Signal", font_size=20, color=RED)
        defect_label.next_to(defect_axes, UP)
        
        # Plot defect signal
        defect_points = []
        for i, val in enumerate(defect_signal[::4]):
            defect_points.append(defect_axes.c2p(i*4, val))
        
        defect_curve = VMobject(color=RED)
        defect_curve.set_points_smoothly(defect_points)
        
        # Highlight defect region using SAMPLE POSITIONS for bbox, REAL POSITIONS for text
        defect_highlight = None
        defect_echo_label = None
        if defect_info and defect_info.get('start_sample', 0) > 0:
            start_sample = defect_info['start_sample']  # USE FOR BBOX
            end_sample = defect_info['end_sample']      # USE FOR BBOX
            depth_range = defect_info.get('depth_range', (0, 100))
            
            if start_sample < len(defect_signal) and end_sample <= len(defect_signal):
                # Calculate highlight position and width using SAMPLE POSITIONS
                highlight_width = (end_sample - start_sample) / len(defect_signal) * 5
                highlight_center_x = defect_axes.c2p((start_sample + end_sample) / 2, 0)[0]
                
                defect_highlight = Rectangle(
                    width=highlight_width, height=2.2, 
                    color=YELLOW, fill_opacity=0.2,
                    stroke_width=2
                )
                defect_highlight.move_to([highlight_center_x, 1.5, 0])
                
                # Show ONLY REAL defect position in text (NO NORMALIZED VALUES!)
                defect_echo_label = Text(f"Defect Echo\n{defect_info['start']:.1f}-{defect_info['end']:.1f}mm\nDepth: {depth_range[0]:.1f}-{depth_range[1]:.1f}mm", 
                                       font_size=16, color=YELLOW)
                defect_echo_label.next_to(defect_highlight, DOWN)
        
        # Animation
        self.play(Create(clean_axes), Write(clean_label))
        self.play(Create(clean_curve))
        self.wait(1)
        
        self.play(Create(defect_axes), Write(defect_label))
        self.play(Create(defect_curve))
        self.wait(1)
        
        if defect_highlight:
            self.play(Create(defect_highlight), Write(defect_echo_label))
        self.wait(1)
    
    def show_cnn_processing(self, signal):
        self.clear()
        
        # Title (no animation)
        title = Text("CNN Feature Extraction Process", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Original signal
        original_axes = Axes(
            x_range=[0, len(signal), len(signal)//4],
            y_range=[np.min(signal)-0.1, np.max(signal)+0.1, 0.2],
            x_length=8,
            y_length=1.5
        ).move_to([0, 2.5, 0])
        
        original_label = Text(f"Input Signal ({len(signal)} samples)", font_size=18)
        original_label.next_to(original_axes, UP)
        
        # Plot original signal
        original_points = []
        for i, val in enumerate(signal[::4]):
            original_points.append(original_axes.c2p(i*4, val))
        
        original_curve = VMobject(color=WHITE)
        original_curve.set_points_smoothly(original_points)
        
        self.play(Create(original_axes), Write(original_label))
        self.play(Create(original_curve))
        self.wait(1)
        
        # Show convolution layers with better positioning
        # Layer 1: 1 -> 32 channels
        layer1_box = Rectangle(width=6, height=0.8, color=GREEN, fill_opacity=0.3)
        layer1_box.move_to([0, 1, 0])
        layer1_label = Text("Conv1D: 1→32 channels, kernel=3", font_size=16)
        layer1_label.move_to(layer1_box.get_center())
        
        # Layer 2: 32 -> 64 channels
        layer2_box = Rectangle(width=5, height=0.8, color=GREEN, fill_opacity=0.5)
        layer2_box.move_to([0, 0, 0])
        layer2_label = Text("Conv1D: 32→64 channels, kernel=3", font_size=16)
        layer2_label.move_to(layer2_box.get_center())
        
        # Layer 3: 64 -> 64 channels (with pooling)
        layer3_box = Rectangle(width=4, height=0.8, color=GREEN, fill_opacity=0.7)
        layer3_box.move_to([0, -1, 0])
        layer3_label = Text("Conv1D: 64→64 channels + Pooling", font_size=16)
        layer3_label.move_to(layer3_box.get_center())
        
        # Feature maps visualization
        feature_maps = VGroup()
        for i in range(8):  # Show 8 feature maps
            # Simulate feature map by applying different filters to original signal
            if i < 4:
                # Edge detection type filters
                feature_signal = np.diff(signal, prepend=signal[0])
                feature_signal = np.maximum(0, feature_signal * (i+1))
            else:
                # Pattern detection type filters
                feature_signal = signal * np.random.uniform(0.5, 2.0) 
                feature_signal = np.maximum(0, feature_signal + np.random.normal(0, 0.05, len(signal)))
            
            feature_axes = Axes(
                x_range=[0, len(feature_signal), len(feature_signal)//4],
                y_range=[0, np.max(feature_signal)+0.1, np.max(feature_signal)/2],
                x_length=1.8,
                y_length=0.7
            ).move_to([-3.5 + i*0.9, -2.5, 0])
            
            # Plot feature
            feature_points = []
            for j, val in enumerate(feature_signal[::8]):
                feature_points.append(feature_axes.c2p(j*8, val))
            
            feature_curve = VMobject(color=YELLOW)
            feature_curve.set_points_smoothly(feature_points)
            
            feature_maps.add(feature_axes, feature_curve)
        
        feature_label = Text("64 Feature Maps (showing 8)", font_size=18)
        feature_label.move_to([0, -3.2, 0])
        
        # Animation with proper arrows
        arrow1 = Arrow(start=[0, 1.8, 0], end=[0, 1.4, 0])
        arrow2 = Arrow(start=[0, 0.6, 0], end=[0, 0.4, 0])
        arrow3 = Arrow(start=[0, -0.4, 0], end=[0, -0.6, 0])
        arrow4 = Arrow(start=[0, -1.4, 0], end=[0, -1.8, 0])
        
        self.play(Create(arrow1))
        self.play(Create(layer1_box), Write(layer1_label))
        self.wait(1)
        
        self.play(Create(arrow2))
        self.play(Create(layer2_box), Write(layer2_label))
        self.wait(1)
        
        self.play(Create(arrow3))
        self.play(Create(layer3_box), Write(layer3_label))
        self.wait(1)
        
        self.play(Create(arrow4))
        self.play(Create(feature_maps), Write(feature_label))
        self.wait(2)
    
    def show_transformer_attention(self, loader):
        self.clear()
        
        # Title (no animation)
        title = Text("Transformer Attention Mechanism", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Get real sequence of 50 signals with mixed defect/clean
        mixed_sequence = loader.get_mixed_sequence_for_attention(seq_length=50)
        if not mixed_sequence:
            print("Could not find mixed sequence for attention")
            return
        
        signals = mixed_sequence['signals']
        labels = mixed_sequence['labels']
        
        # Show subset of signals for visualization (first 12)
        display_count = 12
        sequence_signals = VGroup()
        signal_labels = VGroup()
        
        subtitle = Text(f"Real Training Sequence: 50 signals ({sum(labels)} defects, {50-sum(labels)} clean)", 
                       font_size=18)
        subtitle.move_to([0, 3, 0])
        self.add(subtitle)
        
        for i in range(display_count):
            signal = signals[i]
            
            # Create signal visualization
            signal_axes = Axes(
                x_range=[0, len(signal), len(signal)//4],
                y_range=[np.min(signal), np.max(signal), (np.max(signal)-np.min(signal))/2],
                x_length=1,
                y_length=0.7
            ).move_to([-5.5 + i*1, 2, 0])
            
            # Plot signal
            signal_points = []
            for j, val in enumerate(signal[::16]):
                signal_points.append(signal_axes.c2p(j*16, val))
            
            if labels[i] == 1:  # Defect signal
                signal_curve = VMobject(color=RED, stroke_width=2)
            else:  # Clean signal
                signal_curve = VMobject(color=BLUE, stroke_width=2)
            
            signal_curve.set_points_smoothly(signal_points)
            
            sequence_signals.add(signal_axes, signal_curve)
            
            # Label
            label_text = f"S{i+1}"
            if labels[i] == 1:
                label_text += "D"  # D for defect
            label = Text(label_text, font_size=12)
            label.next_to(signal_axes, DOWN)
            signal_labels.add(label)
        
        # Add indication of remaining signals
        remaining_text = Text(f"... + {50-display_count} more signals", font_size=14)
        remaining_text.move_to([4, 2, 0])
        
        self.play(Create(sequence_signals), Write(signal_labels))
        self.play(Write(remaining_text))
        self.wait(1)
        
        # Show attention weights
        attention_title = Text("Self-Attention: Signal Relationships (12×12 subset)", font_size=18)
        attention_title.move_to([0, 0.5, 0])
        self.play(Write(attention_title))
        
        # Attention matrix visualization based on real labels (12x12 subset)
        attention_matrix = VGroup()
        
        # Create realistic attention weights based on actual defect positions
        attention_weights = np.eye(display_count) * 0.8  # Self-attention baseline
        
        # Add cross-attention between defect signals
        defect_indices = [i for i in range(display_count) if labels[i] == 1]
        for i in defect_indices:
            for j in defect_indices:
                if i != j:
                    attention_weights[i, j] = 0.7  # Strong attention between defects
        
        # Add some attention to neighboring signals
        for i in range(display_count):
            for j in range(display_count):
                if abs(i - j) == 1:  # Adjacent signals
                    attention_weights[i, j] = max(attention_weights[i, j], 0.3)
                elif abs(i - j) == 2:  # Near neighbors
                    attention_weights[i, j] = max(attention_weights[i, j], 0.2)
        
        for i in range(display_count):
            for j in range(display_count):
                cell = Rectangle(width=0.22, height=0.22)
                cell.move_to([-1.21 + j*0.22, -0.2 - i*0.22, 0])
                
                # Color based on attention weight
                opacity = attention_weights[i, j]
                if labels[i] == 1 and labels[j] == 1 and i != j:  # Defect-defect attention
                    cell.set_fill(RED, opacity=opacity)
                elif labels[i] == 1 or labels[j] == 1:  # Defect-clean attention
                    cell.set_fill(ORANGE, opacity=opacity * 0.7)
                else:
                    cell.set_fill(BLUE, opacity=opacity * 0.5)
                
                attention_matrix.add(cell)
        
        # Highlight strong attention between defect signals
        highlights = VGroup()
        for i in defect_indices:
            for j in defect_indices:
                if i != j and attention_weights[i, j] > 0.6:
                    highlight = Rectangle(width=0.26, height=0.26, color=YELLOW, stroke_width=2)
                    highlight.move_to([-1.21 + j*0.22, -0.2 - i*0.22, 0])
                    highlights.add(highlight)
        
        matrix_label = Text("Attention Matrix (12×12 subset)", font_size=14)
        matrix_label.move_to([0, -2.8, 0])
        
        self.play(Create(attention_matrix), Write(matrix_label))
        self.wait(1)
        
        if len(highlights) > 0:
            self.play(Create(highlights))
            
            # Add explanation
            explanation = Text(f"Defect signals attend to each other\n(Real sequence: {len(defect_indices)} defects shown)", 
                              font_size=16, color=YELLOW)
            explanation.move_to([4, -1.5, 0])
            self.play(Write(explanation))
        
        self.wait(3)


class PositionPredictionVisualization(Scene):
    def construct(self):
        # Title (no animation) - CHANGED TITLE
        title = Text("Defect Localization Process", font_size=32)
        title.to_edge(UP)
        self.add(title)
        
        # Load real data and get the SAME signal used in signal comparison
        loader, filename = get_real_data_sample()
        if not loader:
            return
        
        # Get the SAME defect signal used in signal comparison
        clean_signal, defect_signal, defect_info = loader.get_signal_comparison()
        if defect_signal is None or not defect_info:
            return
        
        depth_range = defect_info.get('depth_range', (0, 100))
        
        # Show both signals for comparison - MOVED HIGHER
        signal_axes = Axes(
            x_range=[0, len(defect_signal), len(defect_signal)//4],
            y_range=[np.min(defect_signal)-0.1, np.max(defect_signal)+0.1, 0.2],
            x_length=8,
            y_length=1.5
        ).move_to([0, 1.5, 0])
        
        # signal_label = Text(f"Defect Signal (Depth: {depth_range[0]:.1f}-{depth_range[1]:.1f}mm)",
        #                    font_size=16)
        # signal_label.next_to(signal_axes, UP)
        
        # Add depth range labels at axis ends
        depth_start_label = Text(f"{depth_range[0]:.1f}mm", font_size=14)
        depth_start_label.move_to([-4, 0.7, 0])
        depth_end_label = Text(f"{depth_range[1]:.1f}mm", font_size=14)
        depth_end_label.move_to([4, 0.7, 0])
        
        # Plot defect signal
        defect_points = []
        for i, val in enumerate(defect_signal[::4]):
            defect_points.append(signal_axes.c2p(i*4, val))
        
        defect_curve = VMobject(color=RED, stroke_width=3)
        defect_curve.set_points_smoothly(defect_points)
        
        # Add health signal for comparison - DASHED LINE
        health_axes = Axes(
            x_range=[0, len(clean_signal), len(clean_signal)//4],
            y_range=[np.min(clean_signal)-0.1, np.max(clean_signal)+0.1, 0.2],
            x_length=8,
            y_length=1.5
        ).move_to([0, 1.5, 0])
        
        # Plot health signal with dashed line
        health_points = []
        for i, val in enumerate(clean_signal[::4]):
            health_points.append(health_axes.c2p(i*4, val))
        
        health_curve = VMobject(color=BLUE, stroke_width=2)
        health_curve.set_points_smoothly(health_points)
        health_curve.set_stroke(color=BLUE, width=2, opacity=0.7)
        # Make it dashed by creating multiple small segments
        health_curve_dashed = DashedVMobject(health_curve, num_dashes=50)
        
        health_label = Text("Health Signal", font_size=14, color=BLUE)
        health_label.move_to([3, 2.3, 0])
        
        defect_label = Text("Defect Signal", font_size=14, color=RED)
        defect_label.move_to([3, 2.0, 0])
        
        # True defect position using SAMPLE POSITIONS for bbox
        true_start_sample = defect_info['start_sample']
        true_end_sample = defect_info['end_sample']
        
        true_region = Rectangle(
            width=(true_end_sample - true_start_sample) / len(defect_signal) * 8,
            height=1.7,
            color=GREEN,
            fill_opacity=0.2,
            stroke_width=2
        )
        true_region.move_to([signal_axes.c2p((true_start_sample + true_end_sample)/2, 
                                           (np.min(defect_signal) + np.max(defect_signal))/2)[0], 1.5, 0])
        
        # Position true label right below the true region box
        true_label = Text(f"True: {defect_info['start']:.1f}-{defect_info['end']:.1f}mm", 
                         font_size=14, color=GREEN)
        true_region_center_x = signal_axes.c2p((true_start_sample + true_end_sample)/2, 0)[0]
        true_label.move_to([true_region_center_x, 0.5, 0])
        
        self.play(Create(signal_axes))
        self.play(Write(depth_start_label), Write(depth_end_label))
        self.play(Create(defect_curve), Write(defect_label))
        self.play(Create(health_curve_dashed), Write(health_label))
        self.play(Create(true_region), Write(true_label))
        self.wait(1)
        
        # Show dual position heads with better positioning - MOVED DOWN
        fine_head_box = Rectangle(width=2.5, height=1.5, color=YELLOW, fill_opacity=0.3)
        fine_head_box.move_to([-2.8, -0.8, 0])
        fine_label = Text("FINE HEAD\nLooks at small signal\npatterns for exact\ndefect boundaries", font_size=14, line_spacing=1.15)
        fine_label.move_to(fine_head_box.get_center())
        
        coarse_head_box = Rectangle(width=2.5, height=1.5, color=GOLD, fill_opacity=0.3)
        coarse_head_box.move_to([2.8, -0.8, 0])
        coarse_label = Text("COARSE HEAD\nLooks at broad signal\ntrends for general\ndefect location", font_size=14, line_spacing=1.15)
        coarse_label.move_to(coarse_head_box.get_center())
        
        # Simulate realistic predictions based on true positions with small errors
        depth_span = depth_range[1] - depth_range[0]
        
        # Add small errors in absolute space
        fine_pred_start_abs = defect_info['start'] + np.random.normal(0, depth_span * 0.02)
        fine_pred_end_abs = defect_info['end'] + np.random.normal(0, depth_span * 0.02)
        
        coarse_pred_start_abs = defect_info['start'] + np.random.normal(0, depth_span * 0.05)
        coarse_pred_end_abs = defect_info['end'] + np.random.normal(0, depth_span * 0.05)
        
        # Clamp to depth range
        fine_pred_start_abs = max(depth_range[0], min(depth_range[1], fine_pred_start_abs))
        fine_pred_end_abs = max(depth_range[0], min(depth_range[1], fine_pred_end_abs))
        coarse_pred_start_abs = max(depth_range[0], min(depth_range[1], coarse_pred_start_abs))
        coarse_pred_end_abs = max(depth_range[0], min(depth_range[1], coarse_pred_end_abs))
        
        # Show ONLY REAL positions in text (NO NORMALIZED VALUES!) - MOVED DOWN
        fine_pred = Text(f"Fine: [{fine_pred_start_abs:.1f}, {fine_pred_end_abs:.1f}]mm", 
                        font_size=12, color=YELLOW)
        fine_pred.move_to([-3, -1.8, 0])
        
        coarse_pred = Text(f"Coarse: [{coarse_pred_start_abs:.1f}, {coarse_pred_end_abs:.1f}]mm", 
                          font_size=12, color=GOLD)
        coarse_pred.move_to([3, -1.8, 0])
        
        # Combined prediction
        combined_start_abs = 0.7 * fine_pred_start_abs + 0.3 * coarse_pred_start_abs
        combined_end_abs = 0.7 * fine_pred_end_abs + 0.3 * coarse_pred_end_abs
        
        # Convert absolute positions back to normalized for sample calculation
        combined_start_norm = (combined_start_abs - depth_range[0]) / depth_span
        combined_end_norm = (combined_end_abs - depth_range[0]) / depth_span
        
        # Convert to sample positions for bbox visualization using NEW function signature
        combined_start_sample, combined_end_sample = loader.calculate_defect_signal_positions(
            combined_start_norm, combined_end_norm, len(defect_signal)
        )
        
        combiner = Rectangle(width=3, height=1, color=RED, fill_opacity=0.3)
        combiner.move_to([0, -2.2, 0])
        combiner_label = Text("Weighted Fusion\n0.7×Fine + 0.3×Coarse", font_size=14)
        combiner_label.move_to(combiner.get_center())
        
        # Show ONLY REAL position in text (NO NORMALIZED VALUES!) - MOVED DOWN
        final_pred = Text(f"Final Prediction: [{combined_start_abs:.1f}, {combined_end_abs:.1f}]mm", 
                         font_size=14, color=RED)
        final_pred.move_to([0, -3.2, 0])
        
        # Show predicted region on signal using SAMPLE POSITIONS for bbox
        pred_region = Rectangle(
            width=(combined_end_sample - combined_start_sample) / len(defect_signal) * 8,
            height=1.7,
            color=RED,
            fill_opacity=0.2,
            stroke_width=2
        )
        pred_region.move_to([signal_axes.c2p((combined_start_sample + combined_end_sample)/2, 
                                           (np.min(defect_signal) + np.max(defect_signal))/2)[0], 1.5, 0])
        
        # Position predicted region text right below its box (avoiding intersection with true label)
        pred_region_center_x = signal_axes.c2p((combined_start_sample + combined_end_sample)/2, 0)[0]
        pred_label = Text(f"Predicted: {combined_start_abs:.1f}-{combined_end_abs:.1f}mm", 
                         font_size=14, color=RED)
        # Position to avoid intersection with true label
        if abs(pred_region_center_x - true_region_center_x) < 2:
            pred_label.move_to([pred_region_center_x, 0.2, 0])  # Higher if too close
        else:
            pred_label.move_to([pred_region_center_x, 0.5, 0])  # Same level as true label
        
        # Arrows with better positioning - thin, dashed, to inner sides
        arrow1 = DashedVMobject(Arrow(start=[0, 0.3, 0], end=[-1.5, -0.5, 0], stroke_width=1))
        arrow2 = DashedVMobject(Arrow(start=[0, 0.3, 0], end=[1.5, -0.5, 0], stroke_width=1))
        arrow3 = DashedVMobject(Arrow(start=[-1.5, -1.2, 0], end=[-0.5, -1.6, 0], stroke_width=1))
        arrow4 = DashedVMobject(Arrow(start=[1.5, -1.2, 0], end=[0.5, -1.6, 0], stroke_width=1))
        
        # Animation
        self.play(Create(arrow1), Create(arrow2))
        self.play(Create(fine_head_box), Write(fine_label))
        self.play(Create(coarse_head_box), Write(coarse_label))
        self.wait(1)
        
        self.play(Write(fine_pred), Write(coarse_pred))
        self.wait(1)
        
        self.play(Create(arrow3), Create(arrow4))
        self.play(Create(combiner), Write(combiner_label))
        self.play(Write(final_pred))
        self.wait(1)
        
        # Show final prediction on signal
        self.play(Create(pred_region), Write(pred_label))
        
        # Calculate IoU using sample positions (correct calculation) - MOVED DOWN
        true_length = true_end_sample - true_start_sample
        pred_length = combined_end_sample - combined_start_sample
        
        intersection_start = max(true_start_sample, combined_start_sample)
        intersection_end = min(true_end_sample, combined_end_sample)
        intersection = max(0, intersection_end - intersection_start)
        
        union = true_length + pred_length - intersection
        iou = intersection / union if union > 0 else 0
        
        iou_text = Text(f"IoU = {iou:.3f}", font_size=18, color=PURPLE)
        iou_text.move_to([3, -2, 0])  # MOVED DOWN
        self.play(Write(iou_text))
        self.wait(3)
