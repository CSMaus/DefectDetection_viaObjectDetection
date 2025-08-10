#!/usr/bin/env python3
"""
Main script to run all PAUT visualization animations
Usage: python run_visualizations.py [scene_name]
"""

import os
import sys
import subprocess

# Global debug control - set to False to disable all debug prints
DEBUG_PRINTS = False

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
    
    if DEBUG_PRINTS:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        if DEBUG_PRINTS:
            result = subprocess.run(cmd, check=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if DEBUG_PRINTS:
            print(f"✅ Successfully generated: {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        if DEBUG_PRINTS:
            print(f"❌ Error running {scene_name}:")
            print(f"   Return code: {e.returncode}")
            # Try to run again with output capture to show error
            try:
                error_result = subprocess.run(cmd, capture_output=True, text=True)
                if error_result.stderr:
                    print(f"   Error output: {error_result.stderr}")
                if error_result.stdout:
                    print(f"   Standard output: {error_result.stdout}")
            except:
                pass
        return False
    except FileNotFoundError:
        if DEBUG_PRINTS:
            print("❌ Manim not found. Please install with: pip install manim")
        return False

def main():
    # Change to visualization directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # REMOVED FAKE DATA GENERATION - USE REAL JSON DATA ONLY!
    print("🎬 Using real PAUT JSON data for visualizations...")
    print("🔍 All debug prints from data loading will be shown below...")
    
    # Available scenes - PAUT ONLY
    scenes = {
        "1": ("signal_processing_animation.py", "RealSignalProcessing", "Real Signal Analysis"),
        "2": ("signal_processing_animation.py", "PositionPredictionVisualization", "Position Prediction Process"),
    }
    
    if len(sys.argv) > 1:
        # Run specific scene
        scene_num = sys.argv[1]
        if scene_num in scenes:
            file_name, scene_name, description = scenes[scene_num]
            print(f"🎬 Generating: {description}")
            run_manim_scene(file_name, scene_name, quality="high_quality")  # High quality
        else:
            print(f"❌ Scene '{scene_num}' not found")
            print("Available scenes:")
            for num, (_, _, desc) in scenes.items():
                print(f"  {num}: {desc}")
    else:
        # Run all scenes
        print("🎬 Generating PAUT signal processing animations in HIGH QUALITY...")
        print("=" * 60)
        
        success_count = 0
        total_count = len(scenes)
        
        for num, (file_name, scene_name, description) in scenes.items():
            print(f"\n[{num}/{total_count}] {description}")
            print("-" * 40)
            
            if run_manim_scene(file_name, scene_name, quality="high_quality"):  # High quality
                success_count += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Results: {success_count}/{total_count} animations generated successfully")
        
        if success_count == total_count:
            print("🎉 All animations generated! Check the 'media' folder for output videos.")
        else:
            print("⚠️  Some animations failed. Check error messages above.")
        
        print("\n📁 Output location: ./media/videos/")
        print("🎥 Video format: MP4 (H.264)")
        print("📐 Resolution: 1920x1080 (HIGH QUALITY)")

if __name__ == "__main__":
    main()
