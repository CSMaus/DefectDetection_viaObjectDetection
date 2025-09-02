from manim import *
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import sys

sys.path.append(os.path.dirname(__file__))
from autogates_func import synth_dscan, row_statistic, gradients_1st_2nd, find_peaks_by_second_derivative

config.renderer = "cairo"
config.disable_caching = True
config.frame_rate = 30

class SimpleAutogates(Scene):
    def construct(self):
        # Generate data
        dscan_data = synth_dscan(height=320, width=301, seed=42)
        mean_signal = row_statistic(dscan_data, mode="mean")
        d1, d2 = gradients_1st_2nd(mean_signal)
        peaks, _, _ = find_peaks_by_second_derivative(mean_signal)
        
        # 1. Show D-scan
        title = Text("D-scan", font_size=48)
        title.to_edge(UP)
        
        # Create heatmap image
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(dscan_data, aspect='auto', cmap='jet')
        ax.axis('off')
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, bbox_inches='tight')
        plt.close()
        
        heatmap = ImageMobject(temp_file.name)
        heatmap.scale(1.5)
        os.unlink(temp_file.name)
        
        self.play(Write(title))
        self.play(FadeIn(heatmap))
        self.wait(2)
        
        # 2. Show mean signal
        self.play(FadeOut(heatmap), FadeOut(title))
        
        axes = Axes(x_range=[0, len(mean_signal), 50], y_range=[0, max(mean_signal), 50])
        signal_graph = axes.plot_line_graph([i for i in range(len(mean_signal))], mean_signal, line_color=YELLOW)
        signal_label = Text("Mean Signal", font_size=36).to_edge(UP)
        
        self.play(Write(signal_label))
        self.play(Create(axes), Create(signal_graph))
        self.wait(2)
        
        # 3. Show derivatives
        self.play(FadeOut(axes), FadeOut(signal_graph), FadeOut(signal_label))
        
        # Second derivative plot
        d2_axes = Axes(x_range=[0, len(d2), 50], y_range=[0, max(d2), max(d2)/10])
        d2_graph = d2_axes.plot_line_graph([i for i in range(len(d2))], d2, line_color=BLUE)
        d2_label = Text("2nd Derivative", font_size=36).to_edge(UP)
        
        # Threshold line
        threshold = max(d2) / 4
        threshold_line = d2_axes.get_horizontal_line(d2_axes.c2p(len(d2), threshold)[1])
        threshold_line.set_color(RED)
        
        self.play(Write(d2_label))
        self.play(Create(d2_axes), Create(d2_graph))
        self.play(Create(threshold_line))
        
        # Show peaks
        for start, end in peaks:
            peak_region = Rectangle(
                width=(end-start) * d2_axes.x_length / len(d2),
                height=d2_axes.y_length,
                fill_color=YELLOW,
                fill_opacity=0.3
            ).move_to(d2_axes.c2p((start+end)/2, max(d2)/2))
            self.play(FadeIn(peak_region))
        
        self.wait(3)
