import subprocess
import sys

def main():
    quality = sys.argv[1] if len(sys.argv) > 1 else "high"
    flag = "-qh" if quality == "high" else "-ql"
    
    cmd = ["manim", flag, "simple_autogates.py", "SimpleAutogates", "--disable_caching"]
    print("Running:", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Done!")
    except subprocess.CalledProcessError:
        print("❌ Failed")

if __name__ == "__main__":
    main()
