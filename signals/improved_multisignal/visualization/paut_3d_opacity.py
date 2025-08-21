# paut_3d_opacity.py
from manim import *
import numpy as np
import os, sys

# repo-local imports
sys.path.append(os.path.dirname(__file__))
from real_data_loader import get_real_data_sample, RealPAUTDataLoader

from manim import config
config.renderer = "opengl"          # GPU path (much faster for lots of 3D points)
config.disable_caching = True       # skip hashing huge groups
config.frame_rate = 30              # 60 → 30 cuts frame work ~in half
# Optional: reduce resolution a bit
config.pixel_width = 1280
config.pixel_height = 720

# ------------------------
# Tunables (safe defaults)
# ------------------------
MAX_BEAMS     = 12       # limit beams rendered (vertical axis)
MAX_SCANS     = 60       # limit scans rendered (x axis)
DEPTH_STRIDE  = 8        # subsample depth to reduce objects (>=1)
SPHERE_R      = 0.06     # sphere radius
OPACITY_MODE  = "cubic"  # 'cubic' | 'quadratic' | 'exp'
OPACITY_FLOOR = 0.02     # skip spheres below this opacity (performance)

class PAUT3DOpacityMap(ThreeDScene):
    def construct(self):
        # ------------------------
        # Load real data
        # ------------------------
        loader, filename = get_real_data_sample()
        if not loader or not filename:
            print("Error: could not load PAUT data")
            return

        # Try user's helper if it exists, else fallback to direct parse
        if hasattr(loader, "get_3d_data_sample"):
            d = loader.get_3d_data_sample(filename, max_beams=MAX_BEAMS)
            beams = sorted(list(d["data"].keys()))[:MAX_BEAMS]
            scans_all = sorted({s for b in beams for s in d["data"][b].keys()})
            scans = scans_all[:MAX_SCANS]
            arr = []
            for b in beams:
                row = []
                for s in scans:
                    sig = np.array(d["data"][b][s]["signal"], dtype=float)
                    row.append(sig)
                arr.append(row)
            vol = np.array(arr, dtype=object)  # ragged safe
        else:
            data = loader.load_json_file(filename)
            beam_ids = sorted(
                data.keys(),
                key=lambda x: int(x.replace("Beam_", "")) if str(x).startswith("Beam_") else str(x)
            )[:MAX_BEAMS]
            first_beam = beam_ids[0]
            scan_keys = sorted(data[first_beam].keys(), key=lambda x: int(x.split('_')[0]))[:MAX_SCANS]
            vol_list = []
            for b in beam_ids:
                row = []
                scans_dict = data[b]
                for sk in scan_keys:
                    sd = scans_dict[sk]
                    if isinstance(sd, list):
                        sig = np.array(sd, dtype=float)
                    elif isinstance(sd, dict) and "signal" in sd:
                        sig = np.array(sd["signal"], dtype=float)
                    else:
                        sig = np.array(sd, dtype=float)
                    row.append(sig)
                vol_list.append(row)
            vol = np.array(vol_list, dtype=object)  # (B,S) of 1D arrays

        # Normalize length along depth, get global max
        min_len = min(len(vol[b][s]) for b in range(len(vol)) for s in range(len(vol[0])))
        D = max(1, min_len // DEPTH_STRIDE)
        B, S = len(vol), len(vol[0])

        dense = np.zeros((B, S, D), dtype=float)
        vmax = 1e-9
        for b in range(B):
            for s in range(S):
                sig = vol[b][s][:min_len]
                samp = sig[::DEPTH_STRIDE][:D]
                dense[b, s, :len(samp)] = np.abs(samp)
                vmax = max(vmax, float(np.max(np.abs(samp))))

        # ------------------------
        # Opacity mapping (define BEFORE using)
        # ------------------------
        def map_opacity(v, v_max, mode=OPACITY_MODE):
            if v_max <= 0:
                return 0.0
            n = max(0.0, min(1.0, v / v_max))
            if mode == "quadratic":
                o = n ** 2
            elif mode == "exp":
                # smoother near 0, sharp near 1
                o = (np.exp(3 * n) - 1.0) / (np.exp(3) - 1.0)
            else:  # "cubic" default
                o = n ** 3
            return float(o)

        # ------------------------
        # Axes, labels, camera start
        # ------------------------
        axes = ThreeDAxes(
            x_range=[0, S, max(1, S // 4)],
            y_range=[0, B, max(1, B // 2)],
            z_range=[0, D, max(1, D // 4)],
            x_length=8, y_length=5, z_length=4
        )
        x_lab = Text("X: Scan index", font_size=22, color=BLUE).rotate(PI/2, axis=UP).next_to(axes.x_axis, DOWN)
        y_lab = Text("Y: Beam index", font_size=22, color=GREEN).rotate(PI/2, axis=RIGHT).next_to(axes.y_axis, LEFT)
        z_lab = Text("Z: Depth sample", font_size=22, color=RED).next_to(axes.z_axis, OUT)
        self.add_fixed_in_frame_mobjects(x_lab, y_lab, z_lab)

        title = Text(f"PAUT 3D Data (opacity ∝ value^{3 if OPACITY_MODE=='cubic' else 2})", font_size=28)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP)

        # Start at theta = 0°, nice elevation
        self.set_camera_orientation(phi=60 * DEGREES, theta=0 * DEGREES)

        # ------------------------
        # Points (build BEFORE animating)
        # ------------------------
        points = VGroup()
        for b in range(B):
            for s in range(S):
                for d in range(D):
                    v = dense[b, s, d]
                    op = map_opacity(v, vmax)
                    if op < OPACITY_FLOOR:
                        continue  # skip near-invisible points (perf)
                    # sphere = Sphere(radius=SPHERE_R)
                    # sphere.set_color(WHITE)
                    # sphere.set_opacity(op)
                    # sphere.move_to(axes.c2p(s + 0.5, b + 0.5, d + 0.5))
                    # points.add(sphere)
                    pt = Dot3D(point=axes.c2p(s + 0.5, b + 0.5, d + 0.5), radius=SPHERE_R, color=WHITE)
                    pt.set_opacity(op)
                    points.add(pt)

        print(f"[PAUT3D] B={B} S={S} D={D} total={B * S * D} drawn={len(points)}")

        # ------------------------
        # Animate: axes, points, fly camera 0° → 120°
        # ------------------------
        self.play(Create(axes))
        self.wait(0.2)
        self.add(points)  # add all at once (fast)
        self.move_camera(theta=120 * DEGREES, focal_distance=8, run_time=4, rate_func=smooth)
        self.wait(0.5)

        '''self.play(Create(axes))
        self.wait(0.5)

        if len(points) > 0:
            CHUNK = max(1, len(points) // 6)
            for i in range(0, len(points), CHUNK):
                # self.play(Create(VGroup(*points[i:i + CHUNK])), run_time=0.6)
                self.add(points)

        self.move_camera(
            theta=120 * DEGREES,     # first third of a full orbit
            focal_distance=8,
            run_time=4,
            rate_func=smooth
        )
        self.wait(1)'''