# plot_one_signal.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('..')
from real_data_loader import get_real_data_sample, RealPAUTDataLoader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=None, help="Signal index (0-based).")
    p.add_argument("--first_defect", action="store_true", help="Plot the first defect signal.")
    p.add_argument("--save", type=str, default=None, help="Optional path to save the figure.")
    args = p.parse_args()

    # try default loader
    try:
        loader, filename = get_real_data_sample()
    except FileNotFoundError:
        loader, filename = None, None

    # fallback path if default not found
    if not loader or not filename or not os.path.exists(filename):
        fallback_path = r"D:\ML_DL_AI_stuff\!!NaWoo\code\DefectDetection_viaObjectDetection\signals\improved_multisignal\json_data"
        print(f"Using fallback path: {fallback_path}")
        if not os.path.exists(fallback_path):
            raise RuntimeError(f"Fallback path does not exist: {fallback_path}")
        loader = RealPAUTDataLoader(fallback_path)
        files = [f for f in os.listdir(fallback_path) if f.endswith(".json")]
        if not files:
            raise RuntimeError(f"No JSON files found in {fallback_path}")
        filename = os.path.join(fallback_path, files[0])

    # load sequence
    sequences = loader.get_defect_sequences(filename, seq_length=50)
    if not sequences:
        raise RuntimeError("No sequences found.")

    seq = max(sequences, key=lambda s: sum(s["labels"]))
    signals, labels, info, depth = seq["signals"], seq["labels"], seq["defect_info"], seq["depth_range"]

    # pick signal
    if args.first_defect:
        try:
            sig_idx = next(i for i, y in enumerate(labels) if y > 0.5)
        except StopIteration:
            sig_idx = 0
    else:
        sig_idx = 0 if args.idx is None else int(args.idx)
        sig_idx = max(0, min(sig_idx, len(signals)-1))

    sig = np.asarray(signals[sig_idx], dtype=float)
    y = 1 if labels[sig_idx] > 0.5 else 0
    c = "red" if y else "blue"

    # plot
    plt.figure(figsize=(6,4))
    x = np.arange(len(sig))
    plt.plot(x, sig, color=c, label=("Defect" if y else "Normal"))
    plt.xlabel("Samples"); plt.ylabel("Amplitude")
    plt.grid(True)
    # plt.title(f"Signal {sig_idx+1} | Depth {depth[0]:.1f}–{depth[1]:.1f} mm")
    # plt.legend()

    d = info[sig_idx]
    if y and d and ("start" in d) and ("end" in d):
        plt.text(0.01, 0.97, f"Defect: {d['start']:.1f}-{d['end']:.1f} mm",
                 transform=plt.gca().transAxes, ha="left", va="top")

    if args.save:
        plt.tight_layout(); plt.savefig(args.save, dpi=300, bbox_inches="tight")
    else:
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
