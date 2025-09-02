from manim import *
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from autogates_func import synth_dscan, row_statistic, gradients_1st_2nd, find_peaks_by_second_derivative

config.renderer = "cairo"
config.disable_caching = True
config.frame_rate = 30
config.pixel_width = 1920
config.pixel_height = 1080

class AutogatesVisualization(Scene):
    def construct(self):
        # Generate synthetic D-scan data
        dscan_data = synth_dscan(height=320, width=301, seed=42)
        
        # Calculate statistics and derivatives
        mean_signal = row_statistic(dscan_data, mode="mean")
        d1, d2 = gradients_1st_2nd(mean_signal)
        peaks, _, _ = find_peaks_by_second_derivative(mean_signal)
        
        # Scene 1: Display D-scan heatmap
        self.show_dscan_heatmap(dscan_data)
        
        # Scene 2: Extract and show mean signal
        self.extract_mean_signal(dscan_data, mean_signal)
        
        # Scene 3: Calculate and show derivatives
        self.show_derivatives(mean_signal, d1, d2)
        
        # Scene 4: Zoom into second derivative and find peaks
        self.find_peaks_with_threshold(d2, peaks)
        
    def show_dscan_heatmap(self, dscan_data):
        """Display the D-scan heatmap with title"""
        import matplotlib.pyplot as plt
        import tempfile
        import os
        
        # Create title
        title = Text("D-scan", font_size=48, color=WHITE)
        title.to_edge(UP, buff=0.5)
        
        # Create matplotlib figure and save as temp file
        fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
        im = ax.imshow(dscan_data, aspect='auto', cmap='jet', origin='upper')
        ax.axis('off')
        plt.tight_layout()
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Create ImageMobject from file
        heatmap = ImageMobject(temp_file.name)
        heatmap.scale(2)
        heatmap.move_to(ORIGIN)
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Animate appearance
        self.play(Write(title), run_time=1)
        self.play(FadeIn(heatmap), run_time=2)
        self.wait(1)
        
        # Store for later use
        self.dscan_heatmap = heatmap
        self.dscan_title = title
        self.dscan_data = dscan_data
        
    def extract_mean_signal(self, dscan_data, mean_signal):
        """Show extraction of mean signal from D-scan"""
        # Clear previous scene
        self.play(FadeOut(self.dscan_heatmap), FadeOut(self.dscan_title), run_time=1)
        
        # Create axes for the mean signal plot
        axes = Axes(
            x_range=[0, len(mean_signal), 50],
            y_range=[mean_signal.min(), mean_signal.max(), 50],
            x_length=8,
            y_length=3,
            axis_config={"color": WHITE}
        )
        axes.move_to(ORIGIN)
        
        # Create mean signal plot
        signal_points = [(i, mean_signal[i]) for i in range(len(mean_signal))]
        signal_graph = axes.plot_line_graph(
            x_values=[p[0] for p in signal_points],
            y_values=[p[1] for p in signal_points],
            line_color=YELLOW,
            stroke_width=3
        )
        
        # Add label
        signal_label = Text("Mean Signal", font_size=36, color=YELLOW)
        signal_label.to_edge(UP, buff=0.5)
        
        self.play(Write(signal_label), run_time=1)
        self.play(Create(axes), run_time=1)
        self.play(Create(signal_graph), run_time=2)
        self.wait(1)
        
        # Store for next scene
        self.mean_axes = axes
        self.mean_graph = signal_graph
        self.mean_label = signal_label
        self.mean_signal = mean_signal
        
    def show_derivatives(self, mean_signal, d1, d2):
        """Show calculation of first and second derivatives"""
        # Clear previous
        self.play(FadeOut(self.mean_axes), FadeOut(self.mean_graph), FadeOut(self.mean_label), run_time=1)
        
        # Create second derivative axes
        d2_axes = Axes(
            x_range=[0, len(d2), 50],
            y_range=[0, d2.max(), d2.max()/10],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE}
        )
        d2_axes.move_to(ORIGIN)
        
        # Create second derivative plot
        d2_points = [(i, d2[i]) for i in range(len(d2))]
        d2_graph = d2_axes.plot_line_graph(
            x_values=[p[0] for p in d2_points],
            y_values=[p[1] for p in d2_points],
            line_color=BLUE,
            stroke_width=3
        )
        
        d2_label = Text("2nd Derivative", font_size=36, color=BLUE)
        d2_label.to_edge(UP, buff=0.5)
        
        self.play(Write(d2_label), run_time=1)
        self.play(Create(d2_axes), run_time=1)
        self.play(Create(d2_graph), run_time=2)
        self.wait(1)
        
        # Store for next scene
        self.d2_axes = d2_axes
        self.d2_graph = d2_graph
        self.d2_label = d2_label
        self.d2_data = d2
        
    def find_peaks_with_threshold(self, d2, peaks):
        """Show peak finding with threshold"""
        # Calculate and show threshold line
        threshold = d2.max() / 4.0
        
        # Create threshold line manually
        threshold_line = Line(
            start=self.d2_axes.c2p(0, threshold),
            end=self.d2_axes.c2p(len(d2), threshold),
            color=RED,
            stroke_width=3
        )
        threshold_line.set_stroke(opacity=0.8)
        
        threshold_label = Text(f"Threshold = max/4 = {threshold:.2f}", 
                             font_size=24, color=RED)
        threshold_label.to_edge(DOWN, buff=0.5)
        
        # Show threshold
        self.play(Create(threshold_line), Write(threshold_label), run_time=2)
        
        # Highlight peaks
        for i, (start_idx, end_idx) in enumerate(peaks):
            # Create rectangle highlighting the peak region
            start_point = self.d2_axes.c2p(start_idx, 0)
            end_point = self.d2_axes.c2p(end_idx, d2.max())
            
            highlight = Rectangle(
                width=abs(end_point[0] - start_point[0]),
                height=abs(end_point[1] - start_point[1]),
                color=YELLOW,
                fill_opacity=0.3,
                stroke_opacity=0.8
            )
            highlight.move_to([
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2,
                0
            ])
            
            self.play(FadeIn(highlight), run_time=0.5)
        
        # Add final label
        peaks_label = Text(f"Found {len(peaks)} Gates", 
                          font_size=32, color=YELLOW)
        peaks_label.to_edge(UP, buff=0.5)
        
        self.play(FadeOut(self.d2_label), Write(peaks_label), run_time=1)
        self.wait(2)
