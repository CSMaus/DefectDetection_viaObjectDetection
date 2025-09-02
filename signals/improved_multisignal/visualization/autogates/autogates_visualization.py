# autogates_visualization.py
from manim import *
import numpy as np
from PIL import Image
from matplotlib import colormaps

# ---- absolute layout anchors (in scene coords) ----
CENTER_Y = -0.1           # vertical center for all three axes
HEAT_X   = -5.4           # D-scan center x (left column)
MEAN_X   = -0.5           # mean/d1/d2 “evolving” column stays here
D1_X     =  2.8           # first derivative column
D2_X     =  5.7           # second derivative column

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
    def construct(self):
        # ---------- data ----------
        H, W = 150, 250
        dscan = synth_dscan(height=H, width=W, band_rows=(12, 62), seed=7)

        # ---------- D-scan heatmap (narrower) ----------
        img = _array_to_colormap_image(dscan, cmap_name="turbo")
        heat = ImageMobject(img)
        heat.width = 4.5
        heat.to_edge(LEFT, buff=0.7)

        title = Text("D-scan", font_size=32).next_to(heat, UP)
        self.play(FadeIn(heat, shift=DOWN), FadeIn(title, shift=DOWN), run_time=0.9)

        # --- linkage: sweep down the D-scan while we build the mean curve ---
        row_tracker = ValueTracker(0)

        def _row_center_y(r: float):
            top = heat.get_top()
            return top + DOWN * ((r + 0.5) / H) * heat.height

        sweep = always_redraw(lambda:
            Rectangle(
                width=heat.width, height=heat.height / H,
                stroke_color=YELLOW, fill_color=YELLOW, fill_opacity=0.08, stroke_width=2
            ).move_to(_row_center_y(row_tracker.get_value()))
        )
        self.add(sweep)

        # === right-side panels: mean, d1, d2 (all stay visible) ===
        prof = row_statistic(dscan, mode="mean")
        x_max_prof = float(np.max(prof)) * 1.1
        x_step_prof = nice_step(x_max_prof)

        panel_left = heat.get_right() + RIGHT * 0.4
        col_gap = 0.6
        ax_w, ax_h = 2.5, 4.0

        prof_axes = Axes(
            x_range=[0, x_max_prof, x_step_prof],
            y_range=[0, H, max(1, H // 4)],
            x_length=ax_w, y_length=ax_h,
            axis_config={"stroke_width": 2},
        )
        prof_axes.move_to([MEAN_X, CENTER_Y, 0])  # <-- FIXED POSITION
        label_mean = Text("Accumulated mean", font_size=24)
        label_mean.next_to(prof_axes, UP, buff=0.25)
        self.add(label_mean)  # or self.play(FadeIn(label_mean, shift=UP, run_time=0.3))

        pts_prof = [prof_axes.c2p(float(prof[y]), float(y)) for y in range(H)]
        prof_curve = VMobject(color=WHITE, stroke_width=3).set_points_smoothly(pts_prof[:2])

        # Dot that tracks sweep on the mean curve
        dot = Dot(color=YELLOW, radius=0.06)
        def _dot_updater(mob):
            y = int(np.clip(row_tracker.get_value(), 0, H - 1))
            mob.move_to(prof_axes.c2p(float(prof[y]), float(y)))
        dot.add_updater(_dot_updater)

        self.play(Create(prof_axes), run_time=0.5)
        self.add(dot)

        # Animate sweep + grow mean curve
        for y in range(2, H + 1, max(1, H // 18)):
            row_tracker.set_value(min(H - 1, y - 1))
            prof_curve.set_points_smoothly(pts_prof[:y])
            self.play(Create(prof_curve.copy().set_opacity(0.0)), run_time=0.05)
        self.play(Create(prof_curve), run_time=0.3)
        prof_group = VGroup(prof_axes, prof_curve)

        # 1st derivative (separate x-range, same y)
        d1 = gradient_1d(prof)
        x_max_d1 = max(1e-6, float(np.max(np.abs(d1))) * 1.1)
        d1_axes = Axes(
            x_range=[-x_max_d1, x_max_d1, nice_step(2 * x_max_d1)],
            y_range=[0, H, max(1, H // 4)],
            x_length=ax_w, y_length=ax_h,
            axis_config={"stroke_width": 2},
        )
        d1_axes.move_to([D1_X, CENTER_Y, 0])  # <-- FIXED POSITION
        label_d1 = Text("1st derivative", font_size=24)  # Text("Gradient (1st)"
        label_d1.next_to(d1_axes, UP, buff=0.25)
        self.add(label_d1)

        d1_curve = VMobject(color=BLUE, stroke_width=3).set_points_smoothly(
            [d1_axes.c2p(float(d1[y]), float(y)) for y in range(H)]
        )
        self.play(prof_group.animate.scale(0.95).shift(LEFT * 0.15), run_time=0.4)
        self.play(Create(d1_axes),  run_time=1.2, rate_func=linear)
        self.play(Create(d1_curve),  run_time=1.6, rate_func=linear)

        # 2nd derivative (clip negatives to 0)
        d2 = np.clip(gradient_1d(d1), 0, None)
        x_max_d2 = max(1e-6, float(np.max(d2)) * 1.1)
        d2_axes = Axes(
            x_range=[0, x_max_d2, nice_step(x_max_d2)],
            y_range=[0, H, max(1, H // 4)],
            x_length=ax_w, y_length=ax_h,
            axis_config={"stroke_width": 2},
        )
        d2_axes.move_to([D2_X, CENTER_Y, 0])  # <-- FIXED POSITION
        label_d2 = Text("2nd derivative", font_size=24)
        label_d2.next_to(d2_axes, UP, buff=0.25)
        self.add(label_d2)

        d2_curve = VMobject(color=YELLOW, stroke_width=3).set_points_smoothly(
            [d2_axes.c2p(float(d2[y]), float(y)) for y in range(H)]
        )

        # Small arrows showing “→ derivative → derivative”
        step_arrow1 = Arrow(
            start=[MEAN_X + ax_w / 2 + 0.25, CENTER_Y + 1.0, 0],
            end=[D1_X - ax_w / 2 - 0.25, CENTER_Y + 1.0, 0],
            buff=0.1, stroke_width=2, color=BLUE_E
        )
        step_arrow2 = Arrow(
            start=[D1_X + ax_w / 2 + 0.25, CENTER_Y - 1.0, 0],
            end=[D2_X - ax_w / 2 - 0.25, CENTER_Y - 1.0, 0],
            buff=0.1, stroke_width=2, color=YELLOW
        )
        self.play(GrowArrow(step_arrow1), GrowArrow(step_arrow2), run_time=0.5)

        self.play(Create(d2_axes), run_time=1.2, rate_func=linear)
        self.play(Create(d2_curve), run_time=1.6, rate_func=linear)

        # Threshold & gates **on the 2nd derivative first**
        thr = float(np.max(d2)) / 4.0 if len(d2) else 0.0
        thr_line = DashedLine(
            start=d2_axes.c2p(thr, 0),
            end=d2_axes.c2p(thr, H),
            color=RED, dash_length=0.18, stroke_width=2,
        )
        self.play(Create(thr_line), run_time=0.9)

        peaks, _, _ = find_peaks_by_second_derivative(prof)

        # Gate boxes on d2 panel
        gate_boxes_d2 = VGroup()
        for (y0, y1) in peaks:
            y_mid = 0.5 * (y0 + y1)
            h = (y1 - y0 + 1) * (d2_axes.y_length / (d2_axes.y_range[1] - d2_axes.y_range[0]))
            band = Rectangle(width=d2_axes.x_length, height=h, stroke_width=0,
                             fill_color=GREEN, fill_opacity=0.52)
            band.move_to(d2_axes.c2p(0, y_mid), aligned_edge=LEFT)
            gate_boxes_d2.add(band)
        self.play(FadeIn(gate_boxes_d2, lag_ratio=0.1), run_time=1.2)

        # Mirror those gates back onto the D-scan
        gate_bands_heat = VGroup()
        for (y0, y1) in peaks:
            band_h = ((y1 - y0 + 1) / H) * heat.height
            y_mid = 0.5 * (y0 + y1)
            y_center = heat.get_top() + DOWN * ((y_mid + 0.5) / H) * heat.height
            band = Rectangle(
                width=heat.width, height=band_h,
                stroke_color=GREEN, stroke_width=2,
                fill_color=GREEN, fill_opacity=0.45
            ).move_to([heat.get_center()[0], y_center[1], 0])
            gate_bands_heat.add(band)

        self.play(TransformFromCopy(gate_boxes_d2, gate_bands_heat), run_time=1.2)
        self.wait(0.4)
        self.remove(sweep)

