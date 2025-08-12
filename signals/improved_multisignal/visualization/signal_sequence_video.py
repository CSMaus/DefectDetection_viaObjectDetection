#!/usr/bin/env python3
"""
Single Video: Real Signal Sequence Visualization
Shows signal plots from one real sequence with defect details
"""

from manim import *
import numpy as np
import sys
import os

# Import real data loader
sys.path.append('..')
from real_data_loader import get_real_data_sample

# Import debug control
try:
    from run_visualizations import DEBUG_PRINTS
except ImportError:
    DEBUG_PRINTS = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINTS is True"""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

class RealSignalSequenceVisualization(Scene):
    """Show real signal sequence plots with defect details"""
    
    def construct(self):
        # Load real data
        loader, filename = get_real_data_sample()
        if not loader:
            debug_print("Error: Could not load real PAUT data")
            return
        
        # Get defect sequences
        sequences = loader.get_defect_sequences(filename, seq_length=50)
        if not sequences:
            debug_print(f"No sequences found in {filename}")
            return
        
        # Find sequence with most defects
        best_sequence = None
        max_defects = 0
        
        for sequence in sequences:
            defect_count = sum(sequence['labels'])
            if defect_count > max_defects:
                max_defects = defect_count
                best_sequence = sequence
        
        if not best_sequence:
            best_sequence = sequences[0]
            max_defects = sum(best_sequence['labels'])
        
        debug_print(f"Using sequence with {max_defects} defects")
        
        # Title
        title = Text("Real Signal Sequence: 50 Signals", font_size=32)
        title.to_edge(UP)
        self.add(title)
        
        # Get data
        signals = best_sequence['signals']
        labels = best_sequence['labels']
        defect_info = best_sequence['defect_info']
        depth_range = best_sequence['depth_range']
        
        # Show first 4 signals
        self.show_first_signals(signals, labels, defect_info, depth_range)
        
        # Show dots
        self.show_middle_dots(labels)
        
        # Show last 2 signals
        self.show_last_signals(signals, labels, defect_info, depth_range)
        
        # Show summary
        self.show_sequence_summary(labels, depth_range)
    
    def show_first_signals(self, signals, labels, defect_info, depth_range):
        """Show first 4 signals with details"""
        debug_print("Showing first 4 signals")
        
        signal_plots = VGroup()
        
        for i in range(4):
            signal_data = signals[i]
            is_defect = labels[i] > 0.5
            info = defect_info[i]
            
            # Create axes
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
                defect_text = Text(
                    f"DEFECT\n{info['start']:.1f}-{info['end']:.1f}mm",
                    font_size=12, color=RED
                )
                defect_text.next_to(axes, RIGHT)
                signal_plots.add(axes, signal_curve, label, defect_text)
            else:
                status_text = Text("NORMAL", font_size=12, color=BLUE)
                status_text.next_to(axes, RIGHT)
                signal_plots.add(axes, signal_curve, label, status_text)
        
        self.play(Create(signal_plots), run_time=3)
        self.wait(1)
        
        self.signal_plots = signal_plots
    
    def show_middle_dots(self, labels):
        """Show dots indicating middle signals"""
        debug_print("Showing middle dots")
        
        # Shrink first signals
        self.play(self.signal_plots.animate.scale(0.7).shift(UP * 1), run_time=1)
        
        # Add dots
        dots = VGroup()
        for i in range(3):
            dot = Dot(color=YELLOW, radius=0.1)
            dot.shift(DOWN * (0.5 + i * 0.3))
            dots.add(dot)
        
        # Count middle signals
        middle_defects = sum(labels[4:46])
        middle_normal = 42 - middle_defects
        
        dots_text = Text(
            f"... 42 more signals ...\n{middle_defects} defects, {middle_normal} normal",
            font_size=14, color=YELLOW
        )
        dots_text.next_to(dots, DOWN)
        
        self.play(Create(dots), Write(dots_text), run_time=1)
        self.wait(1)
        
        self.dots = dots
        self.dots_text = dots_text
    
    def show_last_signals(self, signals, labels, defect_info, depth_range):
        """Show last 2 signals"""
        debug_print("Showing last 2 signals")
        
        last_signals = VGroup()
        
        for i, signal_idx in enumerate([48, 49]):
            signal_data = signals[signal_idx]
            is_defect = labels[signal_idx] > 0.5
            info = defect_info[signal_idx]
            
            axes = Axes(
                x_range=[0, len(signal_data), len(signal_data)//4],
                y_range=[np.min(signal_data)-0.1, np.max(signal_data)+0.1, 0.2],
                x_length=4,
                y_length=1,
                axis_config={"color": WHITE, "stroke_width": 1, "tip_length": 0.02}
            )
            axes.shift(DOWN * (2.5 + i * 1.2))
            
            signal_curve = axes.plot(
                lambda x: np.interp(x, np.arange(len(signal_data)), signal_data),
                color=RED if is_defect else BLUE,
                stroke_width=2
            )
            
            label = Text(f"Signal {signal_idx+1}", font_size=14, color=WHITE)
            label.next_to(axes, LEFT)
            
            if is_defect and info.get('start', 0) > 0:
                status_text = f"DEFECT\n{info['start']:.1f}-{info['end']:.1f}mm"
            else:
                status_text = "NORMAL"
            
            status_label = Text(status_text, font_size=10, color=RED if is_defect else BLUE)
            status_label.next_to(axes, RIGHT)
            
            last_signals.add(axes, signal_curve, label, status_label)
        
        self.play(Create(last_signals), run_time=2)
        self.wait(1)
        
        self.last_signals = last_signals
    
    def show_sequence_summary(self, labels, depth_range):
        """Show sequence summary"""
        debug_print("Showing sequence summary")
        
        total_defects = sum(labels)
        total_normal = 50 - total_defects
        
        summary = Text(
            f"Sequence Summary:\n"
            f"Total: 50 signals\n"
            f"Defects: {total_defects}\n"
            f"Normal: {total_normal}\n"
            f"Depth: {depth_range[0]:.1f}-{depth_range[1]:.1f}mm",
            font_size=16, color=WHITE
        )
        summary.to_edge(DOWN)
        
        self.play(Write(summary), run_time=2)
        self.wait(3)
