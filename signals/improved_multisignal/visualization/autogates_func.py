import numpy as np

# -----------------------------
# 1) Synthetic D-scan generator
# -----------------------------
def synth_dscan(
    height=320,
    width=301,
    band_rows=(80, 280),          # rows of the two strong horizontals  
    band_strengths=(1.0, 0.7),    # relative amplitudes of those bands
    band_thickness=3.5,           # Gaussian thickness in rows
    n_defects=4,                   # small bright blobs
    defect_amp_range=(0.35, 0.75),
    defect_sigma_rows=(8.0, 15.0), # vertical size (rows) - bigger areas
    defect_sigma_cols=(8.0, 15.0), # horizontal size (cols) - bigger areas
    vgrad_strength=0.12,          # vertical brightness gradient
    speckle_std=0.06,             # multiplicative speckle (log-normal-ish)
    gaussian_noise_std=0.02,      # additive noise
    seed=42
):
    """
    Returns a 2D float32 array in [0,1] shaped (height, width) approximating your image:
    - two hot horizontal reflectors (red/yellow bands)
    - small roundish defect spots
    - bluish-ish background with vertical gradient + noise
    No plotting is done here.
    """
    rng = np.random.default_rng(seed)
    H, W = int(height), int(width)

    # --- base background with vertical gradient ---
    y = np.linspace(0, 1, H)[:, None]                     # column vector
    vgrad = vgrad_strength * (1.0 - y)                    # brighter near top
    base = 0.25 + 0.2 * rng.standard_normal((H, W)) * 0.0 # flat base level
    img = base + vgrad

    # --- strong horizontal reflectors (Gaussian bands in rows) ---
    yy = np.arange(H)[:, None]
    for r0, amp in zip(band_rows, band_strengths):
        band = amp * np.exp(-0.5 * ((yy - r0) / band_thickness) ** 2)
        img += band

    # --- small bright blobs (defects) as 2D Gaussians ---
    defect_positions = []  # Store defect positions for bottom peak reduction
    for _ in range(n_defects):
        r = rng.integers(low=band_rows[0]+20, high=band_rows[1]-20)  # Only between the two peaks
        c = rng.integers(low=12, high=W-12)
        amp = rng.uniform(*defect_amp_range)
        sr = rng.uniform(*defect_sigma_rows)
        sc = rng.uniform(*defect_sigma_cols)

        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        blob = amp * np.exp(-0.5 * (((yy - r) / sr) ** 2 + ((xx - c) / sc) ** 2))
        img += blob
        
        # Calculate reduction strength based on distance from top peak
        distance_from_top = r - band_rows[0]
        max_distance = band_rows[1] - band_rows[0]
        # Stronger reduction for defects near top (0.8), weaker for bottom (0.3)
        reduction_strength = 0.8 - 0.5 * (distance_from_top / max_distance)
        defect_positions.append((c, sc, amp, reduction_strength))
    
    # Reduce bottom peak amplitude for all defects
    if defect_positions:
        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        for c, sc, defect_amp, strength in defect_positions:
            # Create reduction mask at bottom peak location
            reduction = defect_amp * strength * np.exp(-0.5 * (((yy - band_rows[1]) / band_thickness) ** 2 + ((xx - c) / sc) ** 2))
            img -= reduction

    # --- multiplicative speckle & additive noise ---
    if speckle_std > 0:
        # log-normal like speckle: exp(noise) with small std
        speckle = np.exp(rng.normal(loc=0.0, scale=speckle_std, size=(H, W)))
        img *= speckle
    if gaussian_noise_std > 0:
        img += rng.normal(0.0, gaussian_noise_std, size=(H, W))

    # --- normalize to [0,200] robustly (clip 1st/99th percentiles) ---
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(1e-9, (hi - lo)), 0.0, 1.0) * 200.0
    return img.astype(np.float32)


# ----------------------------------------------------------
# 2) Row statistic along horizontal dimension -> 1-D signal
# ----------------------------------------------------------
def row_statistic(arr2d, mode="mean"):
    """
    Collapses a 2-D array (H x W) along the horizontal dimension (axis=1) into a 1-D array (H,).
    mode ∈ {"mean", "median", "max", "running_max_avg"}:
      - mean:   per-row average
      - median: per-row median
      - max:    per-row maximum
      - running_max_avg: at row y, average of row-wise maxima from row 0..y (accumulated max average)
    """
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


# ---------------------------------------
# 3) Gradient (1st/2nd derivative) in 1D
# ---------------------------------------
def gradient_1d(data):
    """
    Central differences for interior; forward/backward at edges.
    Returns an array with same length as data.
    """
    data = np.asarray(data, dtype=np.float32)
    n = data.size
    if n < 2:
        return np.zeros_like(data)

    grad = np.empty_like(data)
    grad[1:-1] = 0.5 * (data[2:] - data[:-2])
    grad[0] = data[1] - data[0]
    grad[-1] = data[-1] - data[-2]
    return grad

def gradients_1st_2nd(data):
    """Convenience: returns (first_derivative, second_derivative)."""
    d1 = gradient_1d(data)
    d2 = gradient_1d(d1)
    d2 = np.clip(d2, 0, None)  # Set negative values to 0
    return d1, d2


# ----------------------------------------------------
# 4) Peak finding via second-derivative thresholding
# ----------------------------------------------------
def find_peaks_by_second_derivative(data):
    """
    Replicates your logic (without bilateral filtering):
    - compute d1, d2
    - threshold = max(d2) / 4
    - collect regions where d2 >= threshold
    - pair consecutive regions into (start, end) peak intervals
    Returns: (peaks_list, d1, d2)
        peaks_list = list of (start_idx, end_idx)
    """
    data = np.asarray(data, dtype=np.float32)
    d1 = gradient_1d(data)
    d2 = gradient_1d(d1)
    d2 = np.clip(d2, 0, None)  # Set negative values to 0

    thr = float(np.max(d2)) / 4.0 if d2.size > 0 else 0.0

    above_regions = []
    in_region = False
    region_start = 0

    for i, v in enumerate(d2):
        if v >= thr:
            if not in_region:
                in_region = True
                region_start = i
        else:
            if in_region:
                above_regions.append((region_start, i - 1))
                in_region = False
    if in_region:
        above_regions.append((region_start, len(d2) - 1))

    peaks = []
    for i in range(0, len(above_regions) - 1, 2):
        start = above_regions[i][0]
        end = above_regions[i + 1][1]
        peaks.append((start, end))

    return peaks, d1, d2