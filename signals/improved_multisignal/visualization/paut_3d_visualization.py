from manim import *
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from real_data_loader import get_real_data_sample

class PAUT3DStructure(ThreeDScene):
    def construct(self):
        # Load real PAUT data
        loader, filename = get_real_data_sample()
        if not loader:
            print("Error: Could not load real PAUT data")
            return
        
        data_3d = loader.get_3d_data_sample(filename, max_beams=8)
        
        # Title (no animation, appears immediately)
        title = Text("PAUT 3D Data Structure", font_size=36, color=WHITE)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Set camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        
        # Create 3D coordinate system
        max_beams = len(data_3d['data'])
        max_scans = max([len(data_3d['data'][b]) for b in data_3d['data']])
        signal_length = data_3d['dimensions'][2]
        
        axes = ThreeDAxes(
            x_range=[0, max_scans, max(1, max_scans//4)],
            y_range=[0, max_beams, max(1, max_beams//2)],
            z_range=[0, signal_length//10, signal_length//20],
            x_length=8,
            y_length=5,
            z_length=4
        )
        
        # Axis labels (BIGGER TEXT)
        x_label = Text("X: Scan Position", font_size=24, color=BLUE)  # Increased from 20
        y_label = Text("Y: Beam Index", font_size=24, color=GREEN)    # Increased from 20
        z_label = Text("Z: Signal Depth", font_size=24, color=RED)    # Increased from 20
        
        x_label.rotate(PI/2, axis=UP).next_to(axes.x_axis, DOWN)
        y_label.rotate(PI/2, axis=RIGHT).next_to(axes.y_axis, LEFT)
        z_label.next_to(axes.z_axis, OUT)
        
        self.add_fixed_in_frame_mobjects(x_label, y_label, z_label)
        
        # Create data visualization using real signals
        data_points = VGroup()
        defect_highlights = VGroup()
        
        for beam_idx in data_3d['data']:
            for scan_idx in data_3d['data'][beam_idx]:
                signal_data = data_3d['data'][beam_idx][scan_idx]
                signal = signal_data['signal']
                scan_key = signal_data['scan_key']
                
                # Calculate signal energy
                energy = np.sum(signal**2) / len(signal)
                
                # Check if this is a defect signal
                is_defect = scan_key.split('_')[1] != "Health"
                
                # Create point
                point = Sphere(radius=0.08)
                point.move_to(axes.c2p(scan_idx, beam_idx, energy * 1000))  # Scale for visibility
                
                if is_defect:
                    point.set_color(RED)
                    point.set_opacity(0.9)
                    
                    # Add defect highlight
                    highlight = Sphere(radius=0.15, color=YELLOW, fill_opacity=0.3)
                    highlight.move_to(axes.c2p(scan_idx, beam_idx, energy * 1000))
                    defect_highlights.add(highlight)
                else:
                    point.set_color(BLUE)
                    point.set_opacity(0.4)
                
                data_points.add(point)
        
        # Animation sequence
        self.play(Create(axes))
        self.wait(1)
        
        # Show data points gradually
        self.play(Create(data_points), run_time=3)
        self.wait(2)
        
        # Highlight defect areas
        if len(defect_highlights) > 0:
            self.play(Create(defect_highlights))
            self.wait(2)
        
        # Rotate view
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(4)
        self.stop_ambient_camera_rotation()


class SignalSequenceExtraction(Scene):
    def construct(self):
        # Load real data
        loader, filename = get_real_data_sample()
        if not loader:
            return
        
        # Title (no animation)
        title = Text("Signal Sequence Extraction for Neural Networks", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Get real defect sequence
        defect_sequence = loader.get_sample_defect_sequence(seq_length=5)
        if not defect_sequence:
            return
        
        signals = defect_sequence['signals']
        labels = defect_sequence['labels']
        
        # Show 2D representation of signal data
        grid = VGroup()
        x_size = len(signals)
        z_samples = min(8, len(signals[0])//40)  # Sample signal for visualization
        cell_size = 0.5
        
        # Create grid representing signal sequence
        for x in range(x_size):
            for z in range(z_samples):
                cell = Rectangle(width=cell_size, height=cell_size)
                cell.move_to([x*cell_size - 1, z*cell_size - 1.5, 0])
                
                # Get signal amplitude at this sample
                sample_idx = z * (len(signals[x]) // z_samples) if z_samples > 0 else 0
                if sample_idx < len(signals[x]):
                    signal_val = signals[x][sample_idx]
                else:
                    signal_val = 0
                
                # Color based on signal strength and defect presence
                if labels[x] == 1:  # Defect signal
                    if signal_val > 0.5:
                        cell.set_fill(RED, opacity=0.9)
                    elif signal_val > 0.3:
                        cell.set_fill(ORANGE, opacity=0.7)
                    else:
                        cell.set_fill(YELLOW, opacity=0.5)
                else:  # Clean signal
                    if signal_val > 0.3:
                        cell.set_fill(BLUE, opacity=0.6)
                    else:
                        cell.set_fill(BLUE, opacity=0.3)
                
                grid.add(cell)
        
        self.play(Create(grid))
        self.wait(1)
        
        # Show sequence highlighting
        sequence_highlight = Rectangle(
            width=cell_size * x_size,
            height=cell_size * z_samples,
            color=YELLOW,
            stroke_width=4,
            fill_opacity=0
        )
        sequence_highlight.move_to([1.5*cell_size - 1, 2.5*cell_size - 1.5, 0])
        
        self.play(Create(sequence_highlight))
        
        # Show extracted sequence with real signal waveforms
        arrow = Arrow(
            start=sequence_highlight.get_right() + [0.3, 0, 0],
            end=[4, 0, 0],
            color=YELLOW
        )
        
        # Plot actual signal waveforms
        extracted_signals = VGroup()
        for i in range(len(signals)):
            signal = signals[i]
            
            # Downsample for visualization
            downsample_factor = max(1, len(signal) // 50)
            signal_downsampled = signal[::downsample_factor]
            x_vals = np.linspace(0, 2, len(signal_downsampled))
            
            # Create signal curve
            points = []
            for x_val, y_val in zip(x_vals, signal_downsampled):
                points.append([x_val + 4.5, y_val + 1.5 - i*0.6, 0])
            
            if len(points) > 1:
                signal_curve = VMobject()
                signal_curve.set_points_smoothly(points)
                
                # Color based on defect presence
                if labels[i] == 1:
                    signal_curve.set_color(RED)
                    signal_curve.set_stroke_width(3)
                else:
                    signal_curve.set_color(BLUE)
                    signal_curve.set_stroke_width(2)
                
                extracted_signals.add(signal_curve)
        
        sequence_label = Text("Real PAUT Signal Sequence", font_size=20)  # Increased from 16
        sequence_label.next_to(extracted_signals, DOWN)
        
        self.play(Create(arrow))
        self.play(Create(extracted_signals), Write(sequence_label))
        self.wait(2)


class NeuralNetworkProcessing(Scene):
    def construct(self):
        # Title (no animation)
        title = Text("Neural Network Processing Pipeline", font_size=32)
        title.to_edge(UP)
        self.add(title)
        
        # Load real data
        loader, filename = get_real_data_sample()
        if not loader:
            return
        
        defect_sequence = loader.get_sample_defect_sequence(seq_length=5)
        if not defect_sequence:
            return
        
        signals = defect_sequence['signals']
        labels = defect_sequence['labels']
        defect_info = defect_sequence['defect_info']
        
        # Input visualization with real signals
        input_group = VGroup()
        input_label = Text("Input: Real PAUT Signals", font_size=20)  # Increased from 16
        input_label.move_to([-5, 2, 0])
        
        # Show actual signal waveforms
        for i in range(len(signals)):
            signal = signals[i]
            signal_downsampled = signal[::16]  # Downsample for visualization
            x_vals = np.linspace(0, 1.5, len(signal_downsampled))
            
            points = []
            for x_val, y_val in zip(x_vals, signal_downsampled):
                points.append([x_val - 5.5, y_val + 1.5 - i*0.4, 0])
            
            if len(points) > 1:
                signal_curve = VMobject()
                signal_curve.set_points_smoothly(points)
                
                # Color defect signals differently
                if labels[i] == 1:
                    signal_curve.set_color(RED)
                    signal_curve.set_stroke_width(3)
                else:
                    signal_curve.set_color(BLUE)
                    signal_curve.set_stroke_width(2)
                
                input_group.add(signal_curve)
        
        input_group.add(input_label)
        
        # Detection Model (HybridBinaryModel)
        detection_model = VGroup()
        
        # CNN layers
        cnn_box = Rectangle(width=1.2, height=2, color=GREEN, fill_opacity=0.3)
        cnn_box.move_to([-2.5, 0, 0])
        cnn_label = Text("CNN\nFeature\nExtraction", font_size=16)  # Increased from 12
        cnn_label.move_to(cnn_box.get_center())
        
        # Transformer
        transformer_box = Rectangle(width=1.2, height=2, color=PURPLE, fill_opacity=0.3)
        transformer_box.move_to([-0.8, 0, 0])
        transformer_label = Text("Transformer\nSequence\nAnalysis", font_size=16)  # Increased from 12
        transformer_label.move_to(transformer_box.get_center())
        
        # Detection head
        detection_head = Rectangle(width=1, height=1.5, color=ORANGE, fill_opacity=0.3)
        detection_head.move_to([0.8, 0, 0])
        detection_head_label = Text("Detection\nHead", font_size=16)  # Increased from 12
        detection_head_label.move_to(detection_head.get_center())
        
        detection_model.add(cnn_box, cnn_label, transformer_box, transformer_label, 
                          detection_head, detection_head_label)
        
        # Detection results using real labels
        results_group = VGroup()
        results_label = Text("Detection Results", font_size=20)  # Increased from 16
        results_label.move_to([3, 1.5, 0])
        
        # Use actual labels to show realistic probabilities
        for i in range(len(labels)):
            circle = Circle(radius=0.15)
            circle.move_to([3, 1 - i*0.3, 0])
            
            # Simulate realistic probabilities based on actual labels
            if labels[i] == 1:  # Actual defect
                prob = np.random.uniform(0.75, 0.95)
                circle.set_fill(RED, opacity=0.8)
                prob_text = Text(f"{prob:.2f}", font_size=14, color=WHITE)  # Increased from 10
            else:  # Actual clean
                prob = np.random.uniform(0.02, 0.15)
                circle.set_fill(GREEN, opacity=0.8)
                prob_text = Text(f"{prob:.2f}", font_size=14, color=WHITE)  # Increased from 10
            
            prob_text.move_to(circle.get_center())
            results_group.add(circle, prob_text)
        
        results_group.add(results_label)
        
        # Localization model (only for defect signals)
        localization_group = VGroup()
        
        # Position regression head
        position_box = Rectangle(width=1.2, height=1, color=YELLOW, fill_opacity=0.3)
        position_box.move_to([5, 0.3, 0])
        position_label = Text("Position\nRegression", font_size=16)  # Increased from 12
        position_label.move_to(position_box.get_center())
        
        # Position results using real defect positions
        position_results = VGroup()
        defect_positions = []
        for i, info in enumerate(defect_info):
            if info['has_defect']:
                # Normalize positions to [0,1] range
                start_norm = info['start'] / 100.0  # Assuming depth in mm
                end_norm = info['end'] / 100.0
                defect_positions.append((start_norm, end_norm))
        
        for i, (start, end) in enumerate(defect_positions[:2]):  # Show max 2
            pos_rect = Rectangle(width=1.5, height=0.2, color=RED, fill_opacity=0.6)
            pos_rect.move_to([5, -0.3 - i*0.4, 0])
            
            pos_text = Text(f"[{start:.2f}, {end:.2f}]", font_size=14, color=WHITE)  # Increased from 10
            pos_text.move_to(pos_rect.get_center())
            
            position_results.add(pos_rect, pos_text)
        
        localization_group.add(position_box, position_label, position_results)
        
        # Arrows (fixed positions)
        arrow1 = Arrow(start=[-4, 0, 0], end=[-3.1, 0, 0], color=WHITE)
        arrow2 = Arrow(start=[-1.9, 0, 0], end=[-1.4, 0, 0], color=WHITE)
        arrow3 = Arrow(start=[-0.2, 0, 0], end=[0.3, 0, 0], color=WHITE)
        arrow4 = Arrow(start=[1.3, 0, 0], end=[2.2, 0, 0], color=WHITE)
        arrow5 = Arrow(start=[3.5, 0.7, 0], end=[4.4, 0.5, 0], color=RED)
        
        # Animation sequence
        self.play(Create(input_group))
        self.wait(1)
        
        self.play(Create(arrow1))
        self.play(Create(cnn_box), Write(cnn_label))
        self.wait(0.5)
        
        self.play(Create(arrow2))
        self.play(Create(transformer_box), Write(transformer_label))
        self.wait(0.5)
        
        self.play(Create(arrow3))
        self.play(Create(detection_head), Write(detection_head_label))
        self.wait(0.5)
        
        self.play(Create(arrow4))
        self.play(Create(results_group))
        self.wait(1)
        
        # Highlight defect signals
        defect_indices = [i for i, label in enumerate(labels) if label == 1]
        if defect_indices:
            defect_highlight = Rectangle(
                width=0.4, height=len(defect_indices)*0.3 + 0.1, 
                color=YELLOW, stroke_width=3, fill_opacity=0
            )
            defect_highlight.move_to([3, 1 - defect_indices[0]*0.3 - (len(defect_indices)-1)*0.15, 0])
            self.play(Create(defect_highlight))
            
            self.play(Create(arrow5))
            self.play(Create(localization_group))
        
        self.wait(2)


class DetailedArchitecture(Scene):
    def construct(self):
        # Title (no animation)
        title = Text("Detailed Neural Network Architectures", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Detection Model Architecture (HybridBinaryModel)
        detection_title = Text("HybridBinaryModel (Detection)", font_size=20, color=BLUE)
        detection_title.move_to([-3, 2.5, 0])
        
        # Input
        input_box = Rectangle(width=0.8, height=0.6, color=BLUE, fill_opacity=0.3)
        input_box.move_to([-5, 1, 0])
        input_text = Text("320×1", font_size=14)  # Increased from 10
        input_text.move_to(input_box.get_center())
        
        # Conv layers with realistic dimensions
        conv_layers = VGroup()
        conv_configs = [("32×320", GREEN), ("64×320", GREEN), ("64×128", GREEN)]
        for i, (size, color) in enumerate(conv_configs):
            layer = Rectangle(width=0.6, height=1.2 - i*0.1, color=color, fill_opacity=0.3)
            layer.move_to([-3.5 + i*0.8, 1, 0])
            text = Text(size, font_size=12)  # Increased from 8
            text.move_to(layer.get_center())
            conv_layers.add(layer, text)
        
        # Shared layers
        shared_box = Rectangle(width=0.8, height=1, color=PURPLE, fill_opacity=0.3)
        shared_box.move_to([-0.5, 1, 0])
        shared_text = Text("128→64", font_size=14)  # Increased from 10
        shared_text.move_to(shared_box.get_center())
        
        # Transformer
        transformer_box = Rectangle(width=1, height=1.2, color=ORANGE, fill_opacity=0.3)
        transformer_box.move_to([1, 1, 0])
        transformer_text = Text("4×Layers\n8 Heads", font_size=13)  # Increased from 9
        transformer_text.move_to(transformer_box.get_center())
        
        # Binary output
        output_box = Rectangle(width=0.6, height=0.8, color=RED, fill_opacity=0.3)
        output_box.move_to([2.5, 1, 0])
        output_text = Text("Binary\nProb", font_size=13)  # Increased from 9
        output_text.move_to(output_box.get_center())
        
        detection_arch = VGroup(detection_title, input_box, input_text, conv_layers, 
                              shared_box, shared_text, transformer_box, transformer_text,
                              output_box, output_text)
        
        # Localization Model Architecture (EnhancedPositionMultiSignalClassifier)
        loc_title = Text("EnhancedPositionModel (Localization)", font_size=20, color=GREEN)
        loc_title.move_to([-3, -0.5, 0])
        
        # Enhanced feature extraction
        enhanced_conv = Rectangle(width=1.2, height=1, color=GREEN, fill_opacity=0.3)
        enhanced_conv.move_to([-4, -2, 0])
        enhanced_text = Text("Enhanced\nConv+BG\nRemoval", font_size=12)  # Increased from 8
        enhanced_text.move_to(enhanced_conv.get_center())
        
        # Spatial transformer
        spatial_transformer = Rectangle(width=1, height=1, color=PURPLE, fill_opacity=0.3)
        spatial_transformer.move_to([-2, -2, 0])
        spatial_text = Text("Spatial\nTransformer", font_size=13)  # Increased from 9
        spatial_text.move_to(spatial_transformer.get_center())
        
        # Dual position heads
        fine_head = Rectangle(width=0.8, height=0.8, color=YELLOW, fill_opacity=0.3)
        fine_head.move_to([0, -1.5, 0])
        fine_text = Text("Fine\nHead", font_size=13)  # Increased from 9
        fine_text.move_to(fine_head.get_center())
        
        coarse_head = Rectangle(width=0.8, height=0.6, color=GOLD, fill_opacity=0.3)
        coarse_head.move_to([0, -2.5, 0])
        coarse_text = Text("Coarse\nHead", font_size=13)  # Increased from 9
        coarse_text.move_to(coarse_head.get_center())
        
        # Combiner
        combiner = Circle(radius=0.3, color=RED, fill_opacity=0.3)
        combiner.move_to([1.5, -2, 0])
        combiner_text = Text("0.7×F\n+0.3×C", font_size=12)  # Increased from 8
        combiner_text.move_to(combiner.get_center())
        
        # Position output
        pos_output = Rectangle(width=0.8, height=0.6, color=RED, fill_opacity=0.3)
        pos_output.move_to([2.8, -2, 0])
        pos_text = Text("[start,\nend]", font_size=13)  # Increased from 9
        pos_text.move_to(pos_output.get_center())
        
        loc_arch = VGroup(loc_title, enhanced_conv, enhanced_text, spatial_transformer, spatial_text,
                         fine_head, fine_text, coarse_head, coarse_text, combiner, combiner_text,
                         pos_output, pos_text)
        
        # Arrows for detection model (fixed positions)
        det_arrows = VGroup()
        arrow_positions = [(-4.6, 1), (-2.9, 1), (-2.1, 1), (-1.3, 1), (0.5, 1), (1.5, 1), (2.1, 1)]
        for i in range(len(arrow_positions)-1):
            arrow = Arrow(start=[arrow_positions[i][0], arrow_positions[i][1], 0],
                         end=[arrow_positions[i+1][0], arrow_positions[i+1][1], 0],
                         buff=0.1, stroke_width=2)
            det_arrows.add(arrow)
        
        # Arrows for localization model (fixed positions)
        loc_arrows = VGroup()
        # Main flow
        arrow1 = Arrow(start=[-3.4, -2, 0], end=[-2.5, -2, 0], buff=0.1)
        arrow2 = Arrow(start=[-1.5, -2, 0], end=[-0.4, -1.5, 0], buff=0.1)
        arrow3 = Arrow(start=[-1.5, -2, 0], end=[-0.4, -2.5, 0], buff=0.1)
        arrow4 = Arrow(start=[0.4, -1.5, 0], end=[1.2, -1.8, 0], buff=0.1)
        arrow5 = Arrow(start=[0.4, -2.5, 0], end=[1.2, -2.2, 0], buff=0.1)
        arrow6 = Arrow(start=[1.8, -2, 0], end=[2.4, -2, 0], buff=0.1)
        
        loc_arrows.add(arrow1, arrow2, arrow3, arrow4, arrow5, arrow6)
        
        # Animation
        self.play(Create(detection_arch))
        self.play(Create(det_arrows))
        self.wait(2)
        
        self.play(Create(loc_arch))
        self.play(Create(loc_arrows))
        self.wait(3)
