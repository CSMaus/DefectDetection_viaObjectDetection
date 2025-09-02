"""
Run Autogates Visualization with Manim
Usage: python run_autogates_vis.py [quality]
"""

import os
import sys
import subprocess

QUALITY_FLAGS = {
    "low": "-ql",
    "medium": "-qm", 
    "high": "-qh",
    "high_quality": "-qh",
    "4k": "-qk",
}

def run_manim_scene(file_name, scene_name, quality="high"):
    flag = QUALITY_FLAGS.get(quality, "-qh")
    cmd = ["manim", flag, file_name, scene_name, "--disable_caching"]
    print("🎬 Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
        print("✅ Done:", scene_name)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Manim failed with return code:", e.returncode)
        return False
    except FileNotFoundError:
        print("❌ Manim not found. Install with: pip install manim")
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    quality = (sys.argv[1] if len(sys.argv) > 1 else "high").lower()

    print("🔍 Autogates Function Visualization")
    print("=" * 40)
    print("Scene: AutogatesVisualization")
    print("Shows complete autogates process:")
    print("1. D-scan heatmap display")
    print("2. Mean signal extraction")
    print("3. 1st and 2nd derivative calculation")
    print("4. Peak detection with threshold")
    print("Quality:", quality)
    print()

    ok = run_manim_scene("autogates_visualization.py", "AutogatesVisualization", quality)
    if ok:
        print("\n🎥 Output in: ./media/videos/autogates_visualization/")
    else:
        print("\n⚠️  Generation failed.")

if __name__ == "__main__":
    main()
