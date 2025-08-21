"""
Run PAUT 3D Opacity Visualization
Usage:
  python run_paut3d_vis.py [quality]
    quality ∈ {low, medium, high, high_quality, 4k}  (default: high_quality)
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

def run_manim_scene(file_name, scene_name, quality="high_quality"):
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
    # cd to this script’s folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    quality = (sys.argv[1] if len(sys.argv) > 1 else "high_quality").lower()

    print("PAUT 3D Opacity Visualization")
    print("=" * 40)
    print("Scene: PAUT3DOpacityMap (camera fly-through 0 → 120°)")
    print("Quality:", quality)

    ok = run_manim_scene("paut_3d_opacity.py", "PAUT3DOpacityMap", quality)
    if ok:
        print("\n🎥 Output in: ./media/videos/paut_3d_opacity/")
    else:
        print("\n⚠️  Generation failed.")

if __name__ == "__main__":
    main()