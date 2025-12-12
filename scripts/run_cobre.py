"""
Run the COBRE graph-aware mixed effects analysis.

Usage (from project root):

    python -m scripts.run_cobre \
        --model-type full \
        --mat-dir data/raw \
        --out-dir results/cobre

This will:
  1. Load COBRE connectivity data and diagnosis labels
  2. Build cell structure (via COBREAnalysis.preprocess())
  3. Fit the graph-aware mixed effects model (Kim, Kessler & Levina, 2023)
  4. Save cell-level effects to CSV
  5. Print a text summary to stdout and save it to a .txt file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.io_cobre import load_cobre
from pipeline.cobre_analysis import COBREAnalysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run graph-aware mixed effects model on COBRE data."
    )

    parser.add_argument(
        "--model-type",
        choices=["full", "cell_only"],
        default="full",
        help="Which EM model to fit: "
             "'full' includes edge-level effects η, "
             "'cell_only' uses the simplified model without η.",
    )
    parser.add_argument(
        "--mat-dir",
        type=str,
        default="data/raw",
        help="Directory containing COBRE .mat files (passed to load_cobre).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/cobre",
        help="Directory where results (CSV / summary) will be saved.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of EM iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance on log-likelihood.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration EM output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mat_dir = Path(args.mat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load COBRE data
    # ------------------------------------------------------------------
    print(f"Loading COBRE data from {mat_dir} ...")
    X_raw, y_labels, healthy_idx, schizo_idx = load_cobre(mat_dir=mat_dir)

    print(
        f"Loaded X_raw.shape = {X_raw.shape}, "
        f"labels: {np.sum(y_labels == 0)} healthy, "
        f"{np.sum(y_labels == 1)} schizo"
    )

    # ------------------------------------------------------------------
    # 2. Set up analysis object and load data
    # ------------------------------------------------------------------
    analysis = COBREAnalysis()
    analysis.load_data(X_raw=X_raw, y_labels=y_labels)

    # ------------------------------------------------------------------
    # 3. Fit graph-aware mixed effects model via EM
    # ------------------------------------------------------------------
    print(
        f"Fitting model_type={args.model_type!r}, "
        f"max_iter={args.max_iter}, tol={args.tol} ..."
    )

    analysis.fit(
        model_type=args.model_type,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=not args.quiet,
    )

    # ------------------------------------------------------------------
    # 4. Extract results
    # ------------------------------------------------------------------
    # 4a. Cell-level fixed effects → CSV
    cell_effects = analysis.get_cell_effects_df()
    df_cells = pd.DataFrame(cell_effects)

    cell_csv = out_dir / f"cobre_cell_effects_{args.model_type}.csv"
    df_cells.to_csv(cell_csv, index=False)
    print(f"Saved cell-level effects to {cell_csv}")

    # 4b. Summary → stdout + text file
    summary_txt = analysis.summary()
    print("\n" + summary_txt + "\n")

    summary_path = out_dir / f"cobre_summary_{args.model_type}.txt"
    summary_path.write_text(summary_txt)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
