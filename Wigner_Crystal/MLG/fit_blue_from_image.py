#!/usr/bin/env python3
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

METHOD : str = "median"
LINSPACE_N : int = 10000
MAX_STEP : int = 300
LAMBDA : float = 2.0
BLUE_DOM : int = 45
SCORE_EXP : float = 1.0
GREEN_PENALTY : float = 0.5
GREEN_THRESH : int = 80
GREEN_DOM : int = 40
ROI_MARGIN_X : int = 40
ROI_MARGIN_Y : int = 50
IMAGE : str = "mu_MLG_1_2.png"
OUT_CSV : str = "mu_MLG_1_2.csv"
LEFT_END : float = 1.0
RIGHT_END : float = 2.0
LEFT_MARGIN : float = 0.04
RIGHT_MARGIN : float = 0.06


@dataclass
class AxisCalibration:
    ax_slope: float
    ax_intercept: float

    def to_data(self, px: np.ndarray) -> np.ndarray:
        return self.ax_slope * px + self.ax_intercept

    def to_px(self, data: np.ndarray) -> np.ndarray:
        return (data - self.ax_intercept) / self.ax_slope


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def compute_green_mask(img: np.ndarray,
                       green_thresh: int,
                       dominance_margin: int) -> np.ndarray:
    """Return boolean mask where green channel dominates sufficiently."""
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    return (g > green_thresh) & (g - np.maximum(r, b) > dominance_margin)


def find_green_lines(img: np.ndarray,
                     green_thresh: int = 80,
                     dominance_margin: int = 40,
                     min_col_count: int = 50,
                     min_row_count: int = 50,
                     top_k_vertical: int = 3,
                     top_k_horizontal: int = 4) -> Tuple[List[int], List[int]]:
    """Detect prominent green vertical and horizontal grid lines by simple color dominance.

    Returns:
        vertical_xs: sorted list of x indices (length ~3)
        horizontal_ys: sorted list of y indices (length ~4)
    """
    green_mask = compute_green_mask(img, green_thresh=green_thresh, dominance_margin=dominance_margin)

    # Column-wise counts for vertical lines
    col_counts = green_mask.sum(axis=0)
    # Smooth slightly to stabilize peaks
    kernel = np.ones(7) / 7.0
    col_counts_s = np.convolve(col_counts, kernel, mode='same')

    # Row-wise counts for horizontal lines
    row_counts = green_mask.sum(axis=1)
    row_counts_s = np.convolve(row_counts, kernel, mode='same')

    # Pick top K columns far apart
    def pick_peaks(arr: np.ndarray, k: int, min_count: int, min_distance: int) -> List[int]:
        idxs = []
        arr_copy = arr.copy()
        for _ in range(k):
            peak = int(np.argmax(arr_copy))
            if arr_copy[peak] < min_count:
                break
            idxs.append(peak)
            # Suppress neighborhood
            left = max(0, peak - min_distance)
            right = min(arr_copy.size, peak + min_distance + 1)
            arr_copy[left:right] = 0
        return sorted(idxs)

    # Heuristic min distances: 3% of width/height
    h, w, _ = img.shape
    min_dx = max(5, int(0.03 * w))
    min_dy = max(5, int(0.03 * h))

    vertical_xs = pick_peaks(col_counts_s, top_k_vertical, min_col_count, min_dx)
    horizontal_ys = pick_peaks(row_counts_s, top_k_horizontal, min_row_count, min_dy)

    if len(vertical_xs) < 3:
        raise RuntimeError(f"Detected {len(vertical_xs)} vertical green lines; expected 3. Try adjusting thresholds.")
    if len(horizontal_ys) < 4:
        raise RuntimeError(f"Detected {len(horizontal_ys)} horizontal green lines; expected 4. Try adjusting thresholds.")

    # Keep exactly the best 3 and 4 by re-picking with k
    vertical_xs = vertical_xs[:3]
    horizontal_ys = horizontal_ys[:4]

    return vertical_xs, horizontal_ys


def fit_axis_calibrations(img: np.ndarray,
                          vertical_xs: List[int],
                          horizontal_ys: List[int]) -> Tuple[AxisCalibration, AxisCalibration]:
    """Fit linear mappings for ν(x) and μ(y).

    - Vertical lines: left to right correspond to ν = [0.0, 0.5, 1.0]
    - Horizontal lines: top to bottom correspond to μ = [20, 10, 0, -10] meV
    """
    # ν calibration
    nus = np.array([LEFT_END, (LEFT_END+RIGHT_END)/2, RIGHT_END], dtype=float)
    xs = np.array(sorted(vertical_xs), dtype=float)
    if xs.size != nus.size:
        raise RuntimeError("Mismatch in number of ν reference lines.")
    ax = np.polyfit(xs, nus, 1)  # ν = ax[0]*x + ax[1]
    nu_cal = AxisCalibration(ax_slope=ax[0], ax_intercept=ax[1])

    # μ calibration
    mus = np.array([20.0, 10.0, 0.0, -10.0], dtype=float)
    ys = np.array(sorted(horizontal_ys), dtype=float)
    if ys.size != mus.size:
        raise RuntimeError("Mismatch in number of μ reference lines.")
    ay = np.polyfit(ys, mus, 1)  # μ = ay[0]*y + ay[1]
    mu_cal = AxisCalibration(ax_slope=ay[0], ax_intercept=ay[1])

    return nu_cal, mu_cal


def extract_blue_pixels(img: np.ndarray,
                        blue_thresh: int = 90,
                        dominance_margin: int = 45) -> np.ndarray:
    """Return Nx2 array of (x,y) for pixels classified as blue.

    Heuristic: blue channel is high and dominates over red and green.
    """
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    blue_mask = (b > blue_thresh) & (b - np.maximum(r, g) > dominance_margin)

    ys, xs = np.where(blue_mask)
    pts = np.stack([xs, ys], axis=1)
    return pts


def compute_blue_score(img: np.ndarray,
                       roi: Tuple[int, int, int, int],
                       blue_dominance_margin: int,
                       score_exponent: float,
                       green_mask: Optional[np.ndarray] = None,
                       green_penalty: float = 0.5) -> np.ndarray:
    """Compute a per-pixel score for 'blueness' inside ROI.

    score = max(0, B - max(R,G) - blue_dominance_margin) ** score_exponent
    Optionally down-weight scores on green grid pixels by (1 - green_penalty).
    """
    x0, x1, y0, y1 = roi
    r = img[y0:y1, x0:x1, 0].astype(np.float32)
    g = img[y0:y1, x0:x1, 1].astype(np.float32)
    b = img[y0:y1, x0:x1, 2].astype(np.float32)
    base = b - np.maximum(r, g) - float(blue_dominance_margin)
    base = np.maximum(0.0, base)
    if score_exponent != 1.0:
        base = np.power(base, float(score_exponent))
    if green_mask is not None:
        gm = green_mask[y0:y1, x0:x1].astype(np.float32)
        base = base * (1.0 - float(green_penalty) * gm)
    return base


def dp_trace_max_path(score: np.ndarray,
                      max_step: int = MAX_STEP,
                      smoothness_lambda: float = LAMBDA) -> np.ndarray:
    """Dynamic-programming path optimization across columns.

    Args:
        score: HxW non-negative score map
        max_step: maximum |Δy| per column
        smoothness_lambda: penalty weight for |Δy|

    Returns:
        y_indices: length-W array of row indices forming the optimal path
    """
    H, W = score.shape
    dp = np.full((H, W), -np.inf, dtype=np.float32)
    prev = np.full((H, W), -1, dtype=np.int32)
    dp[:, 0] = score[:, 0]

    for x in range(1, W):
        prev_col = dp[:, x - 1]
        for y in range(H):
            yl = max(0, y - max_step)
            yr = min(H - 1, y + max_step)
            window = prev_col[yl:yr + 1]
            dy = np.abs(np.arange(yl, yr + 1) - y)
            cand = window - smoothness_lambda * dy
            k = int(np.argmax(cand))
            dp[y, x] = score[y, x] + cand[k]
            prev[y, x] = yl + k

    # Backtrack
    y_end = int(np.argmax(dp[:, -1]))
    y_path = np.zeros(W, dtype=np.int32)
    y_path[-1] = y_end
    for x in range(W - 1, 0, -1):
        y_path[x - 1] = prev[y_path[x], x]
    return y_path


def consolidate_curve_points(pts: np.ndarray,
                             max_gap_px: int = 1,
                             reducer: str = "median") -> np.ndarray:
    """Reduce per-pixel points to one y per x.

    reducer:
      - "median": median of y at each x (robust)
      - "midpoint": (min(y) + max(y)) / 2 at each x（真ん中）
      - "mean": mean of y at each x
    Returns Nx2 array of (x, y_value).
    """
    if pts.size == 0:
        return pts

    # Group by x
    xs = pts[:, 0]
    ys = pts[:, 1]
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    unique_xs = []
    y_vals = []
    i = 0
    n = xs_sorted.size
    while i < n:
        x0 = xs_sorted[i]
        j = i
        bucket_ys: List[int] = []
        while j < n and xs_sorted[j] == x0:
            bucket_ys.append(int(ys_sorted[j]))
            j += 1
        # Reduce y's for this column according to reducer
        unique_xs.append(int(x0))
        by = np.asarray(bucket_ys, dtype=float)
        if reducer == "midpoint":
            y_vals.append(0.5 * (float(np.min(by)) + float(np.max(by))))
        elif reducer == "mean":
            y_vals.append(float(np.mean(by)))
        else:  # median (default)
            y_vals.append(float(np.median(by)))
        i = j

    curve = np.stack([np.array(unique_xs), np.array(y_vals)], axis=1)

    # Optional: remove large jumps by simple filtering (keep monotone-ish progress)
    # Here, perform a moving median smoothing in y
    if curve.shape[0] > 5:
        k = 5
        ys_sm = np.convolve(curve[:, 1], np.ones(k) / k, mode='same')
        curve[:, 1] = ys_sm

    return curve


def fit_mu_vs_nu(nu: np.ndarray, mu: np.ndarray, degree: int = 3) -> np.ndarray:
    coeffs = np.polyfit(nu, mu, degree)
    return coeffs  # highest power first


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="Extract and fit blue curve from image with green grid calibration.")
    parser.add_argument("--image", type=str, default=IMAGE, help="Path to image file")
    parser.add_argument("--degree", type=int, default=3, help="(unused) kept for backward compatibility; using linear interpolation")
    parser.add_argument("--method", type=str, default=METHOD, choices=["dp", "median"], help="Extraction method: dynamic-programming path or per-column median")
    parser.add_argument("--reducer", type=str, default="median", choices=["median", "midpoint", "mean"], help="If method=median, how to reduce multiple y per x")
    parser.add_argument("--blue_thresh", type=int, default=90, help="Blue threshold for median method")
    parser.add_argument("--blue_dom", type=int, default=45, help="Blue dominance margin (B - max(R,G))")
    parser.add_argument("--green_thresh", type=int, default=80, help="Green detection threshold")
    parser.add_argument("--green_dom", type=int, default=40, help="Green dominance margin")
    parser.add_argument("--roi_margin_x", type=int, default=40, help="Horizontal margin (px) around green grid for ROI")
    parser.add_argument("--roi_margin_y", type=int, default=50, help="Vertical margin (px) around green grid for ROI")
    parser.add_argument("--dp_max_step", type=int, default=MAX_STEP, help="Max |Δy| per column in DP")
    parser.add_argument("--dp_lambda", type=float, default=LAMBDA, help="Smoothness weight for |Δy| in DP")
    parser.add_argument("--score_exp", type=float, default=1.0, help="Exponent applied to blue score")
    parser.add_argument("--green_penalty", type=float, default=0.5, help="Down-weight on green grid in score [0..1]")
    parser.add_argument("--save_debug", action="store_true", help="Save debug score map and DP overlay")
    parser.add_argument("--out_csv", type=str, default=OUT_CSV, help="Output CSV path")
    parser.add_argument("--out_plot", type=str, default="mu_fit_plot.png", help="Output plot (ν-μ)")
    parser.add_argument("--out_overlay", type=str, default="mu_fit_overlay.png", help="Output overlay image path")
    args = parser.parse_args()

    # Resolve all paths relative to the script directory to be robust to CWD
    script_dir = Path(__file__).resolve().parent
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = script_dir / image_path

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = script_dir / out_csv

    out_plot = Path(args.out_plot)
    if not out_plot.is_absolute():
        out_plot = script_dir / out_plot

    out_overlay = Path(args.out_overlay)
    if not out_overlay.is_absolute():
        out_overlay = script_dir / out_overlay

    # Ensure output directories exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    out_overlay.parent.mkdir(parents=True, exist_ok=True)

    img = load_image_as_array(image_path)

    # 1) Detect green grid lines
    vx, hy = find_green_lines(img, green_thresh=args.green_thresh, dominance_margin=args.green_dom)
    print(f"Detected vertical x = {vx}")
    print(f"Detected horizontal y = {hy}")

    # 2) Fit ν(x) and μ(y) calibrations
    nu_cal, mu_cal = fit_axis_calibrations(img, vx, hy)
    print(f"ν(x): slope={nu_cal.ax_slope:.6f}, intercept={nu_cal.ax_intercept:.6f}")
    print(f"μ(y): slope={mu_cal.ax_slope:.6f}, intercept={mu_cal.ax_intercept:.6f}")

    # 3) Extract blue curve (ROI + method)
    h, w, _ = img.shape
    x0 = max(0, int(min(vx)) - int(args.roi_margin_x))
    x1 = min(w, int(max(vx)) + int(args.roi_margin_x))
    y0 = max(0, int(min(hy)) - int(args.roi_margin_y))
    y1 = min(h, int(max(hy)) + int(args.roi_margin_y))
    roi = (x0, x1, y0, y1)

    if args.method == "dp":
        green_mask = compute_green_mask(img, args.green_thresh, args.green_dom)
        score = compute_blue_score(
            img,
            roi=roi,
            blue_dominance_margin=args.blue_dom,
            score_exponent=args.score_exp,
            green_mask=green_mask,
            green_penalty=args.green_penalty,
        )
        y_path = dp_trace_max_path(score, max_step=args.dp_max_step, smoothness_lambda=args.dp_lambda)
        xs_local = np.arange(x0, x1, dtype=np.int32)
        ys_local = y0 + y_path.astype(np.int32)
        curve_px = np.stack([xs_local, ys_local], axis=1)

        if args.save_debug:
            dbg_score = script_dir / "score_map.png"
            plt.figure(figsize=(5, 5), dpi=140)
            plt.imshow(score, cmap='viridis', origin='upper')
            plt.title("Blue score (ROI)")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(str(dbg_score))
            plt.close()

            dbg_overlay = script_dir / "dp_path_overlay.png"
            plt.figure(figsize=(5, 5), dpi=140)
            plt.imshow(img[y0:y1, x0:x1, :])
            plt.plot(np.arange(x1 - x0), y_path, c='red', lw=1.5)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(str(dbg_overlay), bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        blue_pts = extract_blue_pixels(img, blue_thresh=args.blue_thresh, dominance_margin=args.blue_dom)
        if blue_pts.size == 0:
            raise RuntimeError("No blue pixels detected. Adjust thresholds.")
        curve_px = consolidate_curve_points(blue_pts, reducer=args.reducer)

    # 4) Convert to (ν, μ)
    x_px = curve_px[:, 0]
    y_px = curve_px[:, 1]
    nu_vals = nu_cal.to_data(x_px)
    mu_vals = mu_cal.to_data(y_px)

    # Retain only in the ν range [0,1] to avoid spurious bits
    keep = (nu_vals >= LEFT_END+LEFT_MARGIN) & (nu_vals <= RIGHT_END-RIGHT_MARGIN)
    x_px = x_px[keep]
    y_px = y_px[keep]
    nu_vals = nu_vals[keep]
    mu_vals = mu_vals[keep]

    # 5) Linear interpolation μ(ν)
    order_idx = np.argsort(nu_vals)
    nu_sorted = nu_vals[order_idx]
    mu_sorted = mu_vals[order_idx]
    print(f"Using linear interpolation over ν ∈ [{nu_sorted[0]:.3f}, {nu_sorted[-1]:.3f}] with {nu_sorted.size} points.")

    # 6) Save CSV
    header = "x_px,y_px,nu,mu"
    data = np.stack([nu_vals, mu_vals], axis=1)
    np.savetxt(str(out_csv), data, delimiter=",", header=header, comments="")
    print(f"Saved: {out_csv}")

    # 7) Plot ν-μ with fit
    # Avoid external LaTeX dependency; ensure mathtext handles simple symbols
    rcParams["text.usetex"] = False
    plt.figure(figsize=(5, 4), dpi=150)
    plt.scatter(nu_vals, mu_vals, s=8, c='tab:blue', label='extracted')
    nu_lin = np.linspace(float(nu_sorted[0]), float(nu_sorted[-1]), LINSPACE_N)
    mu_lin = np.interp(nu_lin, nu_sorted, mu_sorted)
    plt.plot(nu_lin, mu_lin, c='tab:red', lw=2, label='linear interp')
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\mu$ (meV)")
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_plot))
    plt.close()
    print(f"Saved: {out_plot}")

    # 8) Overlay on original image
    # Build interpolated curve in pixel space across observed ν range
    nu_for_overlay = np.linspace(float(nu_sorted[0]), float(nu_sorted[-1]), LINSPACE_N)
    mu_for_overlay = np.interp(nu_for_overlay, nu_sorted, mu_sorted)
    x_overlay = nu_cal.to_px(nu_for_overlay)
    y_overlay = mu_cal.to_px(mu_for_overlay)

    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(img)
    # Extracted points (sparser for speed)
    if x_px.size > 0:
        step = max(1, x_px.size // 2000)
        plt.scatter(x_px[::step], y_px[::step], s=2, c='cyan', label='extracted')
    plt.plot(x_overlay, y_overlay, c='red', lw=2, label='fit',linestyle="-")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.legend(loc='lower right')
    plt.savefig(str(out_overlay), bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {out_overlay}")


if __name__ == "__main__":
    main()


