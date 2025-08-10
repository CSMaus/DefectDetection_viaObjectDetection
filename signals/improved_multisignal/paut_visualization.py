from manim import *
import numpy as np

class PAUTDataVisualization(Scene):
    def construct(self):
        # Title
        title = Text("PAUT 3D Data Structure & Neural Network Processing", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # 1. PAUT 3D Data Cube Visualization
        self.show_paut_3d_structure()
        self.wait(2)
        
        # 2. Signal Sequence Extraction
        self.show_sequence_extraction()
        self.wait(2)
        
        # 3. Neural Network Processing Pipeline
        self.show_nn_pipeline()
        self.wait(2)

    def show_paut_3d_structure(self):
        """Visualize PAUT 3D data structure"""
        self.clear()
        
        # Create 3D coordinate system
        axes = ThreeDAxes(
            x_range=[0, 10, 2],
            y_range=[0, 8, 2], 
            z_range=[0, 6, 2],
            x_length=6,
            y_length=4,
            z_length=3
        )
        
        # Labels for axes
        x_label = Text("X-Axis (Scan Position)", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Y-Axis (Index Position)", font_size=20).next_to(axes.y_axis, LEFT)
        z_label = Text("Z-Axis (Depth/Time)", font_size=20).next_to(axes.z_axis, OUT)
        
        # Create 3D data cube representation
        cube = Cube(side_length=2, fill_opacity=0.3, fill_color=BLUE)
        cube.move_to(axes.c2p(5, 4, 3))
        
        # Add grid lines to show data structure
        grid_lines = VGroup()
        for i in range(5):
            for j in range(4):
                line = Line3D(
                    start=axes.c2p(2*i, 2*j, 0),
                    end=axes.c2p(2*i, 2*j, 6),
                    color=WHITE,
                    stroke_width=1
                )
                grid_lines.add(line)
        
        # Data dimensions text
        dimensions = Text("PAUT Data: [X×Y×Z] = [Positions×Indices×Samples]", font_size=24)
        dimensions.to_edge(DOWN)
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES)
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label), 
            Write(z_label),
            Create(cube),
            Create(grid_lines),
            Write(dimensions)
        )

    def show_sequence_extraction(self):
        """Show how sequences are extracted from 3D data"""
        self.clear()
        
        # 2D representation of 3D data for clarity
        title = Text("Signal Sequence Extraction", font_size=32)
        title.to_edge(UP)
        
        # Create grid representing 3D data slice
        grid = VGroup()
        signals = VGroup()
        
        rows, cols = 8, 12
        cell_size = 0.4
        
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.move_to([j*cell_size - 2, i*cell_size - 1.5, 0])
                
                # Color some cells to represent defects
                if (i == 3 and 4 <= j <= 7) or (i == 5 and 6 <= j <= 9):
                    cell.set_fill(RED, opacity=0.7)
                else:
                    cell.set_fill(BLUE, opacity=0.3)
                
                grid.add(cell)
        
        # Highlight sequence extraction
        sequence_highlight = Rectangle(
            width=cell_size * 5,
            height=cell_size * rows,
            color=YELLOW,
            stroke_width=4
        )
        sequence_highlight.move_to([2*cell_size - 2, -0.1, 0])
        
        # Arrow and extracted sequence
        arrow = Arrow(
            start=sequence_highlight.get_right() + [0.5, 0, 0],
            end=[4, 0, 0],
            color=YELLOW
        )
        
        # Extracted sequence visualization
        extracted_seq = VGroup()
        for i in range(5):
            signal_line = VGroup()
            # Create sample signal waveform
            x_vals = np.linspace(0, 2, 50)
            if i == 1 or i == 3:  # Defect signals
                y_vals = 0.3 * np.sin(3*x_vals) + 0.2 * np.random.random(50) + 0.5 * np.exp(-(x_vals-1)**2/0.1)
            else:  # Normal signals
                y_vals = 0.2 * np.sin(2*x_vals) + 0.1 * np.random.random(50)
            
            points = [np.array([x, y, 0]) for x, y in zip(x_vals, y_vals)]
            signal_curve = VMobject()
            signal_curve.set_points_smoothly(points)
            signal_curve.set_color(RED if (i == 1 or i == 3) else BLUE)
            signal_curve.move_to([5.5, 1.5 - i*0.6, 0])
            signal_curve.scale(0.5)
            
            extracted_seq.add(signal_curve)
        
        sequence_label = Text("Extracted Signal Sequence", font_size=20)
        sequence_label.next_to(extracted_seq, DOWN)
        
        self.play(Write(title))
        self.play(Create(grid))
        self.play(Create(sequence_highlight))
        self.play(Create(arrow))
        self.play(Create(extracted_seq), Write(sequence_label))

    def show_nn_pipeline(self):
        """Visualize neural network processing pipeline"""
        self.clear()
        
        title = Text("Neural Network Processing Pipeline", font_size=32)
        title.to_edge(UP)
        
        # Input signals
        input_signals = VGroup()
        for i in range(5):
            signal = Rectangle(width=1.5, height=0.3, color=BLUE if i not in [1,3] else RED)
            signal.move_to([-5, 1.5 - i*0.4, 0])
            input_signals.add(signal)
        
        input_label = Text("Input Signals", font_size=16)
        input_label.next_to(input_signals, DOWN)
        
        # Detection Model
        detection_model = VGroup()
        
        # CNN layers
        cnn_layers = VGroup()
        for i in range(3):
            layer = Rectangle(width=0.8, height=2, color=GREEN)
            layer.move_to([-2.5 + i*0.3, 0, 0])
            cnn_layers.add(layer)
        
        cnn_label = Text("CNN\nFeature\nExtraction", font_size=12)
        cnn_label.next_to(cnn_layers, DOWN)
        
        # Transformer layers
        transformer = Rectangle(width=1, height=2, color=PURPLE)
        transformer.move_to([-0.5, 0, 0])
        
        transformer_label = Text("Transformer\nLayers", font_size=12)
        transformer_label.next_to(transformer, DOWN)
        
        # Detection head
        detection_head = Rectangle(width=0.6, height=1.5, color=ORANGE)
        detection_head.move_to([1, 0, 0])
        
        detection_label = Text("Detection\nHead", font_size=12)
        detection_label.next_to(detection_head, DOWN)
        
        detection_model.add(cnn_layers, cnn_label, transformer, transformer_label, detection_head, detection_label)
        
        # Detection results
        detection_results = VGroup()
        for i in range(5):
            result = Circle(radius=0.15)
            if i in [1, 3]:
                result.set_fill(RED, opacity=0.8)
                prob_text = Text("0.87", font_size=10).move_to(result.get_center())
            else:
                result.set_fill(GREEN, opacity=0.8)
                prob_text = Text("0.05", font_size=10).move_to(result.get_center())
            
            result.move_to([2.5, 1.5 - i*0.4, 0])
            detection_results.add(result, prob_text)
        
        detection_results_label = Text("Defect\nProbabilities", font_size=12)
        detection_results_label.next_to(detection_results, DOWN)
        
        # Localization Model (only for defect signals)
        localization_model = VGroup()
        
        # Position head
        position_head = Rectangle(width=1, height=1, color=YELLOW)
        position_head.move_to([4.5, 0.5, 0])
        
        position_label = Text("Position\nRegression", font_size=12)
        position_label.next_to(position_head, DOWN)
        
        # Position results
        position_results = VGroup()
        defect_positions = [(0.23, 0.31), (0.45, 0.52)]
        for i, (start, end) in enumerate(defect_positions):
            pos_rect = Rectangle(width=1.2, height=0.25, color=RED, fill_opacity=0.6)
            pos_rect.move_to([6, 1.1 - i*0.8, 0])
            
            pos_text = Text(f"[{start:.2f}, {end:.2f}]", font_size=10)
            pos_text.move_to(pos_rect.get_center())
            
            position_results.add(pos_rect, pos_text)
        
        position_results_label = Text("Defect\nPositions", font_size=12)
        position_results_label.next_to(position_results, DOWN)
        
        # Arrows
        arrow1 = Arrow(start=input_signals.get_right(), end=detection_model.get_left(), color=WHITE)
        arrow2 = Arrow(start=detection_model.get_right(), end=detection_results.get_left(), color=WHITE)
        arrow3 = Arrow(start=[2.5, 0.7, 0], end=position_head.get_left(), color=RED)
        arrow4 = Arrow(start=position_head.get_right(), end=position_results.get_left(), color=WHITE)
        
        # Animation sequence
        self.play(Write(title))
        self.play(Create(input_signals), Write(input_label))
        self.play(Create(arrow1))
        self.play(Create(detection_model))
        self.play(Create(arrow2))
        self.play(Create(detection_results), Write(detection_results_label))
        self.play(Create(arrow3))
        self.play(Create(localization_model), Write(position_label))
        self.play(Create(arrow4))
        self.play(Create(position_results), Write(position_results_label))


class NetworkArchitecture(Scene):
    def construct(self):
        """Detailed neural network architecture visualization"""
        title = Text("Neural Network Architecture Details", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Detection Model Architecture
        self.show_detection_architecture()
        self.wait(3)
        
        # Localization Model Architecture  
        self.show_localization_architecture()
        self.wait(3)

    def show_detection_architecture(self):
        """Show HybridBinaryModel architecture"""
        self.clear()
        
        subtitle = Text("HybridBinaryModel - Detection Architecture", font_size=24)
        subtitle.to_edge(UP)
        
        # Input layer
        input_layer = Rectangle(width=0.5, height=3, color=BLUE)
        input_layer.move_to([-6, 0, 0])
        input_text = Text("Input\n320×1", font_size=10).next_to(input_layer, DOWN)
        
        # Conv layers
        conv_layers = VGroup()
        conv_configs = [(32, "32×320"), (64, "64×320"), (64, "64×128")]
        for i, (channels, size) in enumerate(conv_configs):
            layer = Rectangle(width=0.6, height=2.5 - i*0.3, color=GREEN)
            layer.move_to([-4 + i*1.2, 0, 0])
            text = Text(f"Conv1D\n{size}", font_size=9).next_to(layer, DOWN)
            conv_layers.add(layer, text)
        
        # Shared layers
        shared_layers = VGroup()
        shared_configs = [128, 64]
        for i, size in enumerate(shared_configs):
            layer = Rectangle(width=0.8, height=2 - i*0.3, color=PURPLE)
            layer.move_to([-0.5 + i*1, 0, 0])
            text = Text(f"Linear\n{size}", font_size=9).next_to(layer, DOWN)
            shared_layers.add(layer, text)
        
        # Transformer
        transformer = Rectangle(width=1, height=2.5, color=ORANGE)
        transformer.move_to([2, 0, 0])
        transformer_text = Text("Transformer\n4 Layers", font_size=10).next_to(transformer, DOWN)
        
        # Output
        output = Rectangle(width=0.6, height=1, color=RED)
        output.move_to([4, 0, 0])
        output_text = Text("Binary\nClassifier", font_size=10).next_to(output, DOWN)
        
        # Arrows
        arrows = VGroup()
        positions = [(-5.5, 0), (-3.2, 0), (-2, 0), (-0.8, 0), (0.5, 0), (1.5, 0), (3, 0)]
        for i in range(len(positions)-1):
            arrow = Arrow(start=[positions[i][0], 0, 0], end=[positions[i+1][0], 0, 0])
            arrows.add(arrow)
        
        self.play(Write(subtitle))
        self.play(Create(input_layer), Write(input_text))
        self.play(Create(conv_layers))
        self.play(Create(shared_layers))
        self.play(Create(transformer), Write(transformer_text))
        self.play(Create(output), Write(output_text))
        self.play(Create(arrows))

    def show_localization_architecture(self):
        """Show EnhancedPositionModel architecture"""
        self.clear()
        
        subtitle = Text("EnhancedPositionModel - Localization Architecture", font_size=24)
        subtitle.to_edge(UP)
        
        # Similar structure but with dual position heads
        # Input processing (same as detection)
        input_layer = Rectangle(width=0.5, height=3, color=BLUE)
        input_layer.move_to([-6, 0, 0])
        
        # Feature extraction
        feature_extraction = Rectangle(width=2, height=2.5, color=GREEN)
        feature_extraction.move_to([-4, 0, 0])
        feature_text = Text("Enhanced\nFeature\nExtraction", font_size=10).next_to(feature_extraction, DOWN)
        
        # Transformer
        transformer = Rectangle(width=1.5, height=2.5, color=ORANGE)
        transformer.move_to([-1, 0, 0])
        transformer_text = Text("Spatial\nTransformer", font_size=10).next_to(transformer, DOWN)
        
        # Dual position heads
        fine_head = Rectangle(width=1, height=1.5, color=YELLOW)
        fine_head.move_to([2, 0.8, 0])
        fine_text = Text("Fine\nPosition", font_size=10).next_to(fine_head, DOWN)
        
        coarse_head = Rectangle(width=1, height=1, color=GOLD)
        coarse_head.move_to([2, -0.8, 0])
        coarse_text = Text("Coarse\nPosition", font_size=10).next_to(coarse_head, DOWN)
        
        # Combination
        combiner = Circle(radius=0.3, color=RED)
        combiner.move_to([4, 0, 0])
        combiner_text = Text("0.7×Fine +\n0.3×Coarse", font_size=8).next_to(combiner, DOWN)
        
        # Output
        output = Rectangle(width=0.8, height=1.2, color=RED)
        output.move_to([5.5, 0, 0])
        output_text = Text("Position\n[start, end]", font_size=10).next_to(output, DOWN)
        
        # Arrows
        arrow1 = Arrow(start=[-5.5, 0, 0], end=[-5, 0, 0])
        arrow2 = Arrow(start=[-3, 0, 0], end=[-2.25, 0, 0])
        arrow3 = Arrow(start=[-0.25, 0, 0], end=[1.5, 0.8, 0])
        arrow4 = Arrow(start=[-0.25, 0, 0], end=[1.5, -0.8, 0])
        arrow5 = Arrow(start=[3, 0.8, 0], end=[3.7, 0.2, 0])
        arrow6 = Arrow(start=[3, -0.8, 0], end=[3.7, -0.2, 0])
        arrow7 = Arrow(start=[4.3, 0, 0], end=[5.1, 0, 0])
        
        self.play(Write(subtitle))
        self.play(Create(input_layer))
        self.play(Create(arrow1))
        self.play(Create(feature_extraction), Write(feature_text))
        self.play(Create(arrow2))
        self.play(Create(transformer), Write(transformer_text))
        self.play(Create(arrow3), Create(arrow4))
        self.play(Create(fine_head), Write(fine_text))
        self.play(Create(coarse_head), Write(coarse_text))
        self.play(Create(arrow5), Create(arrow6))
        self.play(Create(combiner), Write(combiner_text))
        self.play(Create(arrow7))
        self.play(Create(output), Write(output_text))


# Usage instructions:
# 1. Install manim: pip install manim
# 2. Run: manim -pql paut_visualization.py PAUTDataVisualization
# 3. Run: manim -pql paut_visualization.py NetworkArchitecture
