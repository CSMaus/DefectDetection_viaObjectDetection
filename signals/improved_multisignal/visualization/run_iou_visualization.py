#!/usr/bin/env python3
"""
Standalone IoU Visualization Runner
Creates IoU explanation video independently
Usage: python run_iou_visualization.py
"""

import os
import subprocess
import sys

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
    
    print(f"🎬 Running IoU visualization: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ Successfully generated IoU visualization!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running IoU visualization:")
        print(f"   Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ Manim not found. Please install with: pip install manim")
        return False

def main():
    # Change to visualization directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("🎯 IoU Visualization Generator")
    print("=" * 40)
    
    # Check if detection results exist
    if not os.path.exists("detection_results"):
        print("❌ No detection results found!")
        print("   Run yolo_detector.py first to detect objects")
        return
    
    json_files = [f for f in os.listdir("detection_results") if f.endswith('_detections.json')]
    if not json_files:
        print("❌ No detection JSON files found!")
        print("   Run yolo_detector.py first to detect objects")
        return
    
    print(f"✅ Found detection results: {len(json_files)} files")
    
    # Available IoU scenes
    scenes = [
        ("IoUVisualization", "IoU Explanation with Real Objects"),
        ("IoUInteractiveVisualization", "Interactive IoU Demonstration")
    ]
    
    print("\nAvailable IoU visualizations:")
    for i, (scene_name, description) in enumerate(scenes, 1):
        print(f"  {i}: {description}")
    
    # Let user choose or run all
    choice = input("\nEnter scene number (1-2) or press Enter for all: ").strip()
    
    if choice == "":
        # Run all IoU scenes
        print("\n🎬 Generating all IoU visualizations...")
        success_count = 0
        
        for scene_name, description in scenes:
            print(f"\n📹 Creating: {description}")
            print("-" * 30)
            
            if run_manim_scene("iou_visualization.py", scene_name):
                success_count += 1
        
        print(f"\n✅ Completed: {success_count}/{len(scenes)} IoU visualizations generated")
        
    elif choice in ["1", "2"]:
        # Run specific scene
        scene_idx = int(choice) - 1
        scene_name, description = scenes[scene_idx]
        
        print(f"\n🎬 Generating: {description}")
        run_manim_scene("iou_visualization.py", scene_name)
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
