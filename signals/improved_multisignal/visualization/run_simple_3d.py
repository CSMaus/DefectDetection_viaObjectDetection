"""
Run Simple PAUT 3D Visualization with Manim
Usage: python run_simple_3d.py [quality]
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

    print("🔍 Simple PAUT 3D Visualization")
    print("=" * 40)
    print("Scene: SimplePAUT3D")
    print("Shows PAUT data structure from multiple angles")
    print("Low amplitudes = transparent, High amplitudes = opaque")
    print("Quality:", quality)
    print()

    ok = run_manim_scene("simple_3d_paut.py", "SimplePAUT3D", quality)
    if ok:
        print("\n🎥 Output in: ./media/videos/simple_3d_paut/")
    else:
        print("\n⚠️  Generation failed.")

if __name__ == "__main__":
    main()
