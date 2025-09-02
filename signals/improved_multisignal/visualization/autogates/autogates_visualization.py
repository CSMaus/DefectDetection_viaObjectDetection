# autogates_visualization.py
from manim import *
import numpy as np
from PIL import Image
from matplotlib import colormaps


def nice_step(xmax: float, target_ticks: int = 5) -> float:
    """Nice tick step for Axes (1-2-5 progression)."""
    if xmax <= 0:
        return 1.0
    raw = xmax / max(1, target_ticks)
    mag = 10 ** np.floor(np.log10(raw))
    norm = raw / mag
    if norm < 1.5: base = 1
    elif norm < 3: base = 2
    elif norm < 7: base = 5
    else:          base = 10
    return float(base * mag)


def synth_dscan(
    height=320, width=301, band_rows=(80, 280), band_strengths=(1.0, 0.7),
    band_thickness=3.5, n_defects=4, defect_amp_range=(0.35, 0.75),
    defect_sigma_rows=(8.0, 15.0), defect_sigma_cols=(8.0, 15.0),
    vgrad_strength=0.12, speckle_std=0.06, gaussian_noise_std=0.02, seed=42
):
    rng = np.random.default_rng(seed)
    H, W = int(height), int(width)
    y = np.linspace(0, 1, H)[:, None]
    vgrad = vgrad_strength * (1.0 - y)
    base = 0.25 + 0.2 * rng.standard_normal((H, W)) * 0.0
    img = base + vgrad
    yy = np.arange(H)[:, None]
    for r0, amp in zip(band_rows, band_strengths):
        band = amp * np.exp(-0.5 * ((yy - r0) / band_thickness) ** 2)
        img += band
    defect_positions = []
    for _ in range(n_defects):
        r = rng.integers(low=band_rows[0]+20, high=band_rows[1]-20)
        c = rng.integers(low=12, high=W-12)
        amp = rng.uniform(*defect_amp_range)
        sr = rng.uniform(*defect_sigma_rows)
        sc = rng.uniform(*defect_sigma_cols)
        yy = np.arange(H)[:, None]; xx = np.arange(W)[None, :]
        blob = amp * np.exp(-0.5 * (((yy - r)/sr)**2 + ((xx - c)/sc)**2))
        img += blob
        distance_from_top = r - band_rows[0]
        max_distance = band_rows[1] - band_rows[0]
        reduction_strength = 0.8 - 0.5 * (distance_from_top / max_distance)
        defect_positions.append((c, sc, amp, reduction_strength))
    if defect_positions:
        yy = np.arange(H)[:, None]; xx = np.arange(W)[None, :]
        for c, sc, defect_amp, strength in defect_positions:
            reduction = defect_amp * strength * np.exp(
                -0.5 * (((yy - band_rows[1]) / band_thickness) ** 2 +
                        ((xx - c) / sc) ** 2)
            )
            img -= reduction
    if speckle_std > 0:
        speckle = np.exp(rng.normal(loc=0.0, scale=speckle_std, size=(H, W)))
        img *= speckle
    if gaussian_noise_std > 0:
        img += rng.normal(0.0, gaussian_noise_std, size=(H, W))
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(1e-9, (hi - lo)), 0.0, 1.0) * 200.0
    return img.astype(np.float32)

def row_statistic(arr2d, mode="mean"):
    if arr2d.ndim != 2:
        raise ValueError("arr2d must be 2-D (H x W)")
    if mode == "mean":
        prof = arr2d.mean(axis=1)
    elif mode == "median":
        prof = np.median(arr2d, axis=1)
    elif mode == "max":
        prof = arr2d.max(axis=1)
    elif mode == "running_max_avg":
        row_max = arr2d.max(axis=1)
        csum = np.cumsum(row_max)
        denom = np.arange(1, len(row_max) + 1, dtype=np.float32)
        prof = csum / denom
    else:
        raise ValueError("mode must be one of {'mean','median','max','running_max_avg'}")
    return prof.astype(np.float32)

def gradient_1d(data):
    data = np.asarray(data, dtype=np.float32)
    n = data.size
    if n < 2:
        return np.zeros_like(data)
    g = np.empty_like(data)
    g[1:-1] = 0.5 * (data[2:] - data[:-2])
    g[0] = data[1] - data[0]
    g[-1] = data[-1] - data[-2]
    return g

def gradients_1st_2nd(data):
    d1 = gradient_1d(data)
    d2 = gradient_1d(d1)
    d2 = np.clip(d2, 0, None)
    return d1, d2

def find_peaks_by_second_derivative(data):
    data = np.asarray(data, dtype=np.float32)
    d1 = gradient_1d(data)
    d2 = gradient_1d(d1)
    d2 = np.clip(d2, 0, None)
    thr = float(np.max(d2)) / 4.0 if d2.size > 0 else 0.0
    above_regions, in_region, start = [], False, 0
    for i, v in enumerate(d2):
        if v >= thr:
            if not in_region:
                in_region, start = True, i
        else:
            if in_region:
                above_regions.append((start, i - 1))
                in_region = False
    if in_region:
        above_regions.append((start, len(d2) - 1))
    peaks = []
    for i in range(0, len(above_regions) - 1, 2):
        peaks.append((above_regions[i][0], above_regions[i + 1][1]))
    return peaks, d1, d2
# ============================================================================


def _array_to_colormap_image(a, cmap_name="turbo"):
    """
    Convert a 2D array to an RGB uint8 image using a matplotlib colormap,
    without plotting. Returns a PIL Image.
    """
    a = np.asarray(a, dtype=np.float32)
    # Normalize to 0..1 for color mapping (robust to outliers)
    lo, hi = np.percentile(a, [1, 99])
    an = np.clip((a - lo) / max(1e-12, (hi - lo)), 0, 1)
    cmap = colormaps.get_cmap(cmap_name)  # 'turbo' is a nicer rainbow than 'jet'
    rgb = (cmap(an)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


class AutoGateExplainer(Scene):
    """
    Pipeline animation (no long text):
    1) Show D-scan heatmap (rainbow).
    2) Animate row-wise reduction to 1-D mean profile (right panel).
    3) Transform profile -> 1st derivative.
    4) Transform -> 2nd derivative.
    5) Zoom 2nd derivative, draw 1/4 max threshold, highlight gates.
    """
    def construct(self):
        # ---------- data ----------
        H, W = 80, 300  # as requested: ~300 x 80 (width x height)
        dscan = synth_dscan(height=H, width=W, band_rows=(12, 62), seed=7)

        # ---------- D-scan heatmap ----------
        img = _array_to_colormap_image(dscan, cmap_name="turbo")
        heat = ImageMobject(img)
        heat.width = 5.8
        heat.to_edge(LEFT, buff=0.7)

        title = Text("D-scan", font_size=32)
        title.next_to(heat, UP)

        self.play(FadeIn(heat, shift=DOWN), FadeIn(title, shift=DOWN), run_time=0.9)
        self.wait(0.4)

        # ---------- mean profile axes (right) ----------
        prof = row_statistic(dscan, mode="mean")  # or "running_max_avg" if that’s what you want to show
        x_max = float(np.max(prof)) * 1.1
        x_step = nice_step(x_max)

        prof_axes = Axes(
            x_range=[0, x_max, x_step], # <-- valid positive step
            y_range=[0, H, max(1, H // 4)],
            x_length=4.6, y_length=5.0,
            axis_config={"stroke_width": 2}
        )
        prof_axes.to_edge(RIGHT, buff=0.8)
        self.play(Create(prof_axes), run_time=0.6)

        # Animated “row sweeping” indicator on D-scan
        row_tracker = ValueTracker(0)  # 0 .. H-1

        def row_center_y(r: float):
            top = heat.get_top()
            return top + DOWN * ((r + 0.5) / H) * heat.height

        sweep = always_redraw(
            lambda: Rectangle(
                width=heat.width,
                height=heat.height / H,
                stroke_color=YELLOW,
                fill_color=YELLOW,
                fill_opacity=0.08,
                stroke_width=2,
            ).move_to(row_center_y(row_tracker.get_value()))
        )
        self.add(sweep)

        # one full pass, top → bottom
        self.play(row_tracker.animate.set_value(H - 1), run_time=1.8, rate_func=linear)

        # Compute profile (mean across x)
        prof = row_statistic(dscan, mode="mean")
        x_max = float(np.max(prof)) * 1.1
        # prof_axes.x_axis.set_range(0, x_max, 0.0)  # lead to the errror

        # Build the polyline progressively (y from 0..H-1)
        prof_curve = VMobject(stroke_color=WHITE, stroke_width=3)
        pts = []
        for y in range(H):
            x = prof[y]
            px, py = prof_axes.c2p(x, y)[:2]
            pts.append(np.array([px, py, 0.0]))
        prof_curve.set_points_smoothly(pts[:2])

        # Link a moving dot from sweep to the current profile point
        dot = Dot(color=YELLOW, radius=0.06)
        def dot_updater(mob):
            y = int(self.sweep_row)
            y = max(0, min(H-1, y))
            x = prof[y]
            mob.move_to(prof_axes.c2p(x, y))
        dot.add_updater(dot_updater)

        self.add(dot)

        # Animate sweep + curve growth
        for y in range(2, H + 1, max(1, H // 20)):
            self.sweep_row = min(H - 1, y - 1)
            prof_curve.set_points_smoothly(pts[:y])
            self.play(Create(prof_curve.copy().set_opacity(0.0)), run_time=0.05)  # tick
        self.remove(sweep, dot)
        self.play(Create(prof_curve), run_time=0.4)
        self.wait(0.3)

        # ---------- first derivative ----------
        d1, d2 = gradients_1st_2nd(prof)
        d1_curve = self._curve_from_profile(prof_axes, d1, color=BLUE_E)
        arrow1 = Arrow(start=prof_axes.c2p(x_max*0.8, H*0.85),
                       end=prof_axes.c2p(x_max*0.93, H*0.85),
                       buff=0, stroke_width=3, color=BLUE_E)
        self.play(GrowArrow(arrow1), Transform(prof_curve, d1_curve), run_time=0.8)
        self.wait(0.2)

        # ---------- second derivative ----------
        d2_curve = self._curve_from_profile(prof_axes, d2, color=ORANGE)
        arrow2 = Arrow(start=prof_axes.c2p(x_max*0.8, H*0.15),
                       end=prof_axes.c2p(x_max*0.93, H*0.15),
                       buff=0, stroke_width=3, color=ORANGE)
        self.play(GrowArrow(arrow2), Transform(prof_curve, d2_curve), run_time=0.8)
        self.wait(0.3)

        # ---------- focus on 2nd derivative ----------
        self.play(FadeOut(heat), FadeOut(title), FadeOut(arrow1), FadeOut(arrow2), run_time=0.5)
        self.play(prof_axes.animate.scale(1.1).shift(LEFT*0.8), run_time=0.5)
        self.wait(0.2)

        # Threshold line at 1/4 max(d2)
        thr = float(np.max(d2)) / 4.0 if d2.size else 0.0
        x_thr = min(x_max, thr)
        thr_line = DashedLine(
            start=prof_axes.c2p(x_thr, 0),
            end=prof_axes.c2p(x_thr, H),
            dash_length=0.12, color=YELLOW
        )
        self.play(Create(thr_line), run_time=0.6)

        # Gate regions (using your logic)
        peaks, _, d2_full = find_peaks_by_second_derivative(prof)
        gate_boxes = VGroup()
        for (y0, y1) in peaks:
            r = Rectangle(
                width=prof_axes.x_length,
                height=(y1 - y0) * (prof_axes.y_length / (prof_axes.y_range[1] - prof_axes.y_range[0])),
                stroke_width=0,
                fill_color=GREEN,
                fill_opacity=0.25
            )
            # center at mid y, left-aligned to axes x=0
            y_mid = 0.5 * (y0 + y1)
            r.move_to(prof_axes.c2p(0, y_mid), aligned_edge=LEFT)
            gate_boxes.add(r)

        if len(gate_boxes) > 0:
            self.play(FadeIn(gate_boxes, lag_ratio=0.1), run_time=0.9)
        self.wait(1.0)

    # helper: make a smooth curve for a y-indexed profile
    def _curve_from_profile(self, axes: Axes, profile: np.ndarray, color=WHITE):
        H = len(profile)
        pts = [axes.c2p(float(profile[y]), float(y)) for y in range(H)]
        curve = VMobject(stroke_color=color, stroke_width=3)
        if H >= 2:
            curve.set_points_smoothly(np.array(pts))
        else:
            curve.set_points_as_corners(np.array(pts))
        return curve

