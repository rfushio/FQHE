import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay ~N curves of capacitance vs r_hole for different fixed "
            "(D_Br_TG, lambda) pairs from capacitance_by_params.csv"
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
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance_vs_rhole_overlay.png"
        ),
        help="Path to save the overlay plot figure.",
    )
    parser.add_argument(
        "--n_curves",
        type=int,
        default=5,
        help="Number of (D_Br_TG, lambda) pairs to plot as separate curves.",
    )
    parser.add_argument(
        "--round_decimals",
        type=int,
        default=6,
        help="Decimals used to identify unique parameter pairs robustly to float noise.",
    )
    return parser.parse_args()


def choose_pairs(df: pd.DataFrame, n_curves: int, round_decimals: int) -> List[Tuple[float, float]]:
    df = df.copy()
    df["D_pair"] = df["D_Br_TG"].round(round_decimals)
    df["lambda_pair"] = df["lambda"].round(round_decimals)
    unique_pairs = (
        df[["D_pair", "lambda_pair"]]
        .drop_duplicates()
        .sort_values(["D_pair", "lambda_pair"])  # deterministic order
        .to_records(index=False)
    )
    unique_pairs = [(float(d), float(l)) for d, l in unique_pairs]

    if len(unique_pairs) == 0:
        return []
    if len(unique_pairs) <= n_curves:
        return unique_pairs

    # Evenly spaced selection across the list to get diverse coverage
    idxs = np.linspace(0, len(unique_pairs) - 1, n_curves).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    return [unique_pairs[i] for i in idxs]


def main() -> None:
    args = parse_arguments()
    df = pd.read_csv(args.input)

    for col in ["r_hole", "D_Br_TG", "lambda", "capacitance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["r_hole", "D_Br_TG", "lambda", "capacitance"]).reset_index(drop=True)

    pairs = choose_pairs(df, args.n_curves, args.round_decimals)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for (D_val, lam_val) in pairs:
        mask = (
            np.isclose(
                df["D_Br_TG"],
                D_val,
                rtol=1e-08,
                atol=10 ** (-args.round_decimals),
            )
            & np.isclose(
                df["lambda"],
                lam_val,
                rtol=1e-08,
                atol=10 ** (-args.round_decimals),
            )
        )
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("r_hole")
        ax.plot(
            sub["r_hole"].to_numpy(),
            sub["capacitance"].to_numpy(),
            marker="o",
            lw=1.0,
            ms=4,
            label=f"D={D_val:g} nm, λ={lam_val:g} nm",
        )

    ax.set_title("Capacitance vs r_hole for fixed (D_Br_TG, lambda)")
    ax.set_xlabel("r_hole [nm]")
    ax.set_ylabel("capacitance [F]")
    ax.grid(True, ls=":", alpha=0.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="best", fontsize=8)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=300)


if __name__ == "__main__":
    main()


