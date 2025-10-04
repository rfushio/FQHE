import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot capacitance vs r_hole, D_Br_TG, and lambda from a CSV with "
            "columns r_hole, D_Br_TG, lambda, capacitance."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance_by_params.csv"
        ),
        help="Path to the input CSV containing r_hole, D_Br_TG, lambda, capacitance.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance_relations.png"
        ),
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance_relation_summary.csv"
        ),
        help="Path to save a CSV summary of linear fits (slope, intercept, R^2).",
    )
    return parser.parse_args()


def aggregate_relationship(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(x_col)["capacitance"]
        .agg(["mean", "min", "max", "count"])  # mean for trend, min/max for envelope
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "mean": "capacitance_mean",
            "min": "capacitance_min",
            "max": "capacitance_max",
        }
    )
    grouped = grouped.sort_values(x_col).reset_index(drop=True)
    return grouped


def linear_fit_with_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size < 2 or np.allclose(np.max(x) - np.min(x), 0.0):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def plot_relationship(ax: plt.Axes, data: pd.DataFrame, x_col: str, title: str) -> Dict[str, float]:
    x = data[x_col].to_numpy(dtype=float)
    y = data["capacitance_mean"].to_numpy(dtype=float)
    y_min = data["capacitance_min"].to_numpy(dtype=float)
    y_max = data["capacitance_max"].to_numpy(dtype=float)

    line = ax.plot(x, y, "o-", lw=1.0, ms=4)[0]
    color = line.get_color()
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.2)
    slope, intercept, r2 = linear_fit_with_r2(x, y)

    if np.isfinite(slope):
        xfit = np.linspace(np.min(x), np.max(x), 100)
        yfit = slope * xfit + intercept
        ax.plot(xfit, yfit, "r--", lw=1.0)
        label = f"slope={slope:.3e} F/unit, R^2={r2:.3f}"
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top")

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("capacitance [F]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(True, ls=":", alpha=0.5)

    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": float(x.size)}


def main() -> None:
    args = parse_arguments()
    df = pd.read_csv(args.input)

    for col in ["r_hole", "D_Br_TG", "lambda", "capacitance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["r_hole", "D_Br_TG", "lambda", "capacitance"]).reset_index(drop=True)

    rel_r_hole = aggregate_relationship(df, "r_hole")
    rel_D = aggregate_relationship(df, "D_Br_TG")
    rel_lambda = aggregate_relationship(df, "lambda")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    metrics_summary: Dict[str, Dict[str, float]] = {}
    metrics_summary["r_hole"] = plot_relationship(axes[0], rel_r_hole, "r_hole", "Capacitance vs r_hole")
    metrics_summary["D_Br_TG"] = plot_relationship(axes[1], rel_D, "D_Br_TG", "Capacitance vs D_Br_TG")
    metrics_summary["lambda"] = plot_relationship(axes[2], rel_lambda, "lambda", "Capacitance vs lambda")

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300)

    summary_rows = []
    for key, vals in metrics_summary.items():
        summary_rows.append({
            "parameter": key,
            "slope": vals["slope"],
            "intercept": vals["intercept"],
            "r2": vals["r2"],
            "n_points": int(vals["n_points"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()


