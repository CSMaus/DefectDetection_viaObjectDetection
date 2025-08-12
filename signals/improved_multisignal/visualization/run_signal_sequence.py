#!/usr/bin/env python3
"""
Single Signal Sequence Video Generator
Creates one video showing real signal sequence plots
"""

import os
import subprocess

def run_manim_scene(file_name, scene_name, quality="high_quality"):
    """Run a specific Manim scene"""
    quality_flags = {
        "low": "-ql",
        "medium": "-qm", 
        "high": "-qh",
        "high_quality": "-qh",
        "4k": "-qk"
    }
    
    flag = quality_flags.get(quality, "-qh")
    cmd = ["manim", flag, file_name, scene_name]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"Successfully generated: {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {scene_name}:")
        print(f"   Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("Manim not found. Please install with: pip install manim")
        return False

def main():
    # Change to visualization directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Signal Sequence Video Generator")
    print("=" * 40)
    print("Creating video showing real signal sequence plots")
    print("Using real PAUT data from json_data/")
    print("=" * 40)
    
    # Generate the video
    success = run_manim_scene("signal_sequence_video.py", "RealSignalSequenceVisualization")
    
    if success:
        print("\nVideo generated successfully!")
        print("Output location: ./media/videos/")
    else:
        print("\nVideo generation failed!")

if __name__ == "__main__":
    main()
