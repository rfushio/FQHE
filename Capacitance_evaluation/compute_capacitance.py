import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute capacitance as the slope of Electric displacement vs V_QPC "
            "for each (r_hole, D_Br_TG, lambda) combination."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance2.csv"
        ),
        help="Path to the input COMSOL CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Users/rikutofushio/00ARIP/FQHE/Capacitance_evaluation/capacitance_by_params.csv"
        ),
        help="Path to write the output CSV (r_hole,D_Br_TG,lambda,capacitance).",
    )
    return parser.parse_args()


def read_comsol_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a COMSOL-exported CSV where metadata lines start with '%' and the header
    row may also be commented. We provide explicit column names and skip comment
    lines. Returns a DataFrame with numeric columns.
    """
    column_names: List[str] = [
        "r_hole (nm)",
        "D_Br_TG (nm)",
        "lambda (nm)",
        "V_QPC (V)",
        "Electric displacement field, z component (C)",
        ]

    df = pd.read_csv(
        csv_path,
        comment='%',
        names=column_names,
        skip_blank_lines=True,
    )

    # Coerce all relevant columns to numeric
    for col in column_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing essential values
    df = df.dropna(subset=[
        "V_QPC (V)",
        "r_hole (nm)",
        "D_Br_TG (nm)",
        "lambda (nm)",
        "Electric displacement field, z component (C)",
    ])

    return df


def compute_group_slope(group: pd.DataFrame) -> float:
    """
    Compute slope d(Electric displacement)/d(V_QPC) using a linear fit for a
    single parameter group. Returns NaN if not enough variation in V_QPC.
    """
    x = group["V_QPC (V)"].to_numpy(dtype=float)
    y = group["Electric displacement field, z component (C)"].to_numpy(dtype=float)

    # Filter finite values
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    # Require at least two distinct V values
    if x.size < 2 or np.allclose(np.max(x) - np.min(x), 0.0):
        return float("nan")

    # Linear fit: y = slope * x + intercept
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def compute_capacitance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (r_hole, D_Br_TG, lambda) and compute the slope (capacitance) for
    each group. Returns a tidy DataFrame with columns:
    r_hole, D_Br_TG, lambda, capacitance
    """
    group_keys = ["r_hole (nm)", "D_Br_TG (nm)", "lambda (nm)"]
    slope_series = df.groupby(group_keys, dropna=False).apply(compute_group_slope)
    slope_series.name = "capacitance"

    result = slope_series.reset_index()
    result = result.rename(columns={
        "r_hole (nm)": "r_hole",
        "D_Br_TG (nm)": "D_Br_TG",
        "lambda (nm)": "lambda",
    })

    # Drop groups where slope could not be computed
    result = result.dropna(subset=["capacitance"]).reset_index(drop=True)

    # Sort for readability
    result = result.sort_values(["r_hole", "D_Br_TG", "lambda"]).reset_index(drop=True)
    return result


def main() -> None:
    args = parse_arguments()
    df = read_comsol_csv(args.input)
    result = compute_capacitance(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()


