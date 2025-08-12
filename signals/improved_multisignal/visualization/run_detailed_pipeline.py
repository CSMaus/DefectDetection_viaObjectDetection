#!/usr/bin/env python3
"""
DETAILED Neural Network Pipeline Video Generator
Creates 5 EXTREMELY DETAILED videos showing REAL HybridBinaryModel processing:

1. RealSignalSequenceVisualization - REAL 50-signal sequence with actual defect locations
2. RealFeatureExtractionVisualization - REAL conv layers, pooling, shared layer outputs  
3. RealTransformerInputVisualization - REAL 64-dim feature vectors with actual values
4. RealTransformerProcessingVisualization - REAL attention weights and layer transformations
5. RealBinaryClassificationVisualization - REAL predictions for each signal with confidence

Usage: 
    python run_detailed_pipeline.py              # Generate all 5 videos
    python run_detailed_pipeline.py 1            # Generate specific video (1-5)
    python run_detailed_pipeline.py all          # Generate all videos and combine them
"""

import os
import sys
import subprocess
import argparse

# Global debug control
DEBUG_PRINTS = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINTS is True"""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

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
    
    debug_print(f"Running: {' '.join(cmd)}")
    
    try:
        if DEBUG_PRINTS:
            result = subprocess.run(cmd, check=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        debug_print(f"✅ Successfully generated: {scene_name}")
        return True
    except subprocess.CalledProcessError as e:
        debug_print(f"❌ Error running {scene_name}:")
        debug_print(f"   Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        debug_print("❌ Manim not found. Please install with: pip install manim")
        return False

def combine_videos():
    """Combine all 5 videos into one seamless video"""
    debug_print("🎬 Combining all 5 DETAILED videos into one...")
    
    # Video files in order
    video_files = [
        "media/videos/detailed_neural_pipeline/1080p60/RealSignalSequenceVisualization.mp4",
        "media/videos/detailed_neural_pipeline/1080p60/RealFeatureExtractionVisualization.mp4", 
        "media/videos/detailed_neural_pipeline/1080p60/RealTransformerInputVisualization.mp4",
        "media/videos/detailed_neural_pipeline/1080p60/RealTransformerProcessingVisualization.mp4",
        "media/videos/detailed_neural_pipeline/1080p60/RealBinaryClassificationVisualization.mp4"
    ]
    
    # Check if all videos exist
    missing_videos = []
    for video in video_files:
        if not os.path.exists(video):
            missing_videos.append(video)
    
    if missing_videos:
        debug_print("❌ Missing videos:")
        for video in missing_videos:
            debug_print(f"   {video}")
        debug_print("Generate all videos first before combining.")
        return False
    
    # Create file list for ffmpeg
    with open("detailed_video_list.txt", "w") as f:
        for video in video_files:
            f.write(f"file '{video}'\n")
    
    # Combine videos using ffmpeg
    output_file = "media/videos/complete_detailed_neural_pipeline.mp4"
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "detailed_video_list.txt",
        "-c", "copy", output_file, "-y"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=not DEBUG_PRINTS, text=True)
        debug_print(f"✅ Combined DETAILED video saved to: {output_file}")
        
        # Clean up
        os.remove("detailed_video_list.txt")
        return True
        
    except subprocess.CalledProcessError as e:
        debug_print(f"❌ Error combining videos: {e}")
        return False
    except FileNotFoundError:
        debug_print("❌ ffmpeg not found. Please install ffmpeg to combine videos.")
        return False

def main():
    # Change to visualization directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # DETAILED Neural Network Pipeline scenes
    scenes = {
        "1": ("detailed_neural_pipeline.py", "RealSignalSequenceVisualization", "REAL Signal Sequence (50 signals with actual defects)"),
        "2": ("detailed_neural_pipeline.py", "RealFeatureExtractionVisualization", "REAL Feature Extraction (shared layer processing)"),
        "3": ("detailed_neural_pipeline.py", "RealTransformerInputVisualization", "REAL Transformer Input (feature vectors + positional encoding)"),
        "4": ("detailed_neural_pipeline.py", "RealTransformerProcessingVisualization", "REAL Transformer Processing (layer-by-layer transformations)"),
        "5": ("detailed_neural_pipeline.py", "RealBinaryClassificationVisualization", "REAL Binary Classification (actual predictions)")
    }
    
    parser = argparse.ArgumentParser(description="Generate DETAILED Neural Network Pipeline Videos")
    parser.add_argument("scene", nargs="?", default="all", 
                       help="Scene number (1-5), 'all' for all scenes, or 'combine' to combine existing videos")
    parser.add_argument("--quality", default="high_quality", 
                       choices=["low", "medium", "high", "high_quality", "4k"],
                       help="Video quality")
    
    args = parser.parse_args()
    
    print("🧠 DETAILED Neural Network Pipeline Video Generator")
    print("=" * 70)
    print("Using YOUR REAL ImprovedModel and YOUR REAL training data")
    print("Model: models/improved_model_20250710_193851/best_model.pth")
    print("Data: json_data/ (REAL 50-signal sequences with defects)")
    print("Shows: REAL features, REAL processing, REAL predictions")
    print("=" * 70)
    
    if args.scene == "combine":
        # Just combine existing videos
        combine_videos()
        return
    
    if args.scene == "all":
        # Generate all scenes
        print("🎬 Generating ALL 5 DETAILED neural network pipeline videos...")
        
        success_count = 0
        total_count = len(scenes)
        
        for num, (file_name, scene_name, description) in scenes.items():
            print(f"\n[{num}/{total_count}] {description}")
            print("-" * 60)
            
            if run_manim_scene(file_name, scene_name, args.quality):
                success_count += 1
        
        print("\n" + "=" * 70)
        print(f"📊 Results: {success_count}/{total_count} DETAILED videos generated successfully")
        
        if success_count == total_count:
            print("🎉 All DETAILED videos generated!")
            
            # Ask if user wants to combine them
            try:
                response = input("\n🔗 Combine all videos into one DETAILED pipeline? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    combine_videos()
            except KeyboardInterrupt:
                print("\n👋 Skipping video combination.")
        else:
            print("⚠️  Some videos failed. Check error messages above.")
            
    elif args.scene in scenes:
        # Generate specific scene
        file_name, scene_name, description = scenes[args.scene]
        print(f"🎬 Generating DETAILED Video {args.scene}: {description}")
        run_manim_scene(file_name, scene_name, args.quality)
        
    else:
        print(f"❌ Invalid scene '{args.scene}'")
        print("Available scenes:")
        for num, (_, _, desc) in scenes.items():
            print(f"  {num}: {desc}")
        print("  all: Generate all videos")
        print("  combine: Combine existing videos")
    
    print(f"\n📁 Output location: ./media/videos/")
    print(f"🎥 Video format: MP4 (H.264)")
    print(f"📐 Quality: {args.quality}")
    print(f"🔥 Content: REAL data, REAL model, REAL processing details!")

if __name__ == "__main__":
    main()
