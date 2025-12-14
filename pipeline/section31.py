import numpy as np
from typing import Dict, List
from scipy import stats
from src.model.block_descent import BlockCDConfig, GraphAwareEM_BlockCD


def run_section31_article_style(
    X_235: np.ndarray,          # (N, E) edge weights (Power-235 edges)
    X_cov: np.ndarray,          # (N, p) covariates [1, disease, ...]
    cells: np.ndarray,          # (C, 2) cell definitions (system indices)
    cell_id_of_edge: np.ndarray,# (E,) edge→cell mapping [0,...,C-1]
    fit_max_iter: int = 100,
    significance_level: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Replicate Section 3.1-style hypothesis testing for group comparisons
    using the BlockCD EM and article-style GLS standard errors.

    Steps:
      1. Fit GraphAwareEM_BlockCD with diagonal V.
      2. Compute GLS SEs via model.compute_gls_standard_errors()
         (cell-only GLS formula with Σ = V + Z U Z^T).
      3. Perform Wald tests for disease effect α_1^{(c)} in each cell.
    """
    if verbose:
        print("=" * 70)
        print("Section 3.1 – Article-style analysis (BlockCD EM + GLS SE)")
        print("=" * 70)

    N, E = X_235.shape
    p = X_cov.shape[1]
    C = len(cells)

    # ------------------------------------------------------------------
    # Build edges_per_cell
    # ------------------------------------------------------------------
    edges_per_cell: List[np.ndarray] = [
        np.where(cell_id_of_edge == c)[0] for c in range(C)
    ]

    if verbose:
        print(f"\nData dimensions:")
        print(f"  N = {N} subjects")
        print(f"  E = {E} edges")
        print(f"  C = {C} cells")
        print(f"  p = {p} covariates")

    # ------------------------------------------------------------------
    # Step 1: Fit BlockCD EM
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nStep 1: Fitting GraphAwareEM_BlockCD (max_outer_iter={fit_max_iter}) ...")

    config = BlockCDConfig(
        max_outer_iter=fit_max_iter,
        max_inner_iter=20,
        max_cd_iter=50,
        outer_tol=1e-4,
        inner_tol=1e-6,
        cd_tol=1e-8,
        min_variance=1e-8,
        verbose=verbose,
    )

    model = GraphAwareEM_BlockCD(
        Y=X_235,
        X=X_cov,
        cell_id_of_edge=cell_id_of_edge,
        edges_per_cell=edges_per_cell,
        config=config,
    )
    model.fit()

    alpha = model.get_cell_effects()             # (C, p)
    U = model.get_random_effects_covariance()   # (C, C) – for diagnostics
    V_diag = model.get_residual_variances()     # (E,)

    if verbose:
        print("\nFitted parameters:")
        print(f"  α shape: {alpha.shape}")
        print(f"  U shape: {U.shape}")
        print(f"  V_diag shape: {V_diag.shape}")

    # ------------------------------------------------------------------
    # Step 2: Compute GLS standard errors for α
    # ------------------------------------------------------------------
    if verbose:
        print("\nStep 2: Computing GLS standard errors for α ...")

    se = model.compute_gls_standard_errors()     # (C, p)

    # ------------------------------------------------------------------
    # Step 3: Wald tests for disease effect in each cell
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nStep 3: Wald tests for disease effect (α = {significance_level})")

    # assuming disease covariate is column 1 (0 = intercept)
    disease_idx = 1 if p >= 2 else 0

    alpha1 = alpha[:, disease_idx]              # (C,)
    se1 = se[:, disease_idx]                    # (C,)

    z_stats = alpha1 / se1
    p_values = 2 * stats.norm.sf(np.abs(z_stats))
    significant = p_values < significance_level
    n_significant = int(np.sum(significant))

    if verbose:
        print(f"\n  Disease effect (column {disease_idx}):")
        print(f"    mean α₁: {np.mean(alpha1):+.4f}")
        print(f"    mean SE: {np.mean(se1):.4f}")
        print(f"    mean |z|: {np.mean(np.abs(z_stats)):.3f}")
        print(f"\n  *** Significant cells (p < {significance_level}): {n_significant} ***")
        print("  (Paper reports 13)")

    # ------------------------------------------------------------------
    # Step 4: List significant cells (sorted by p-value)
    # ------------------------------------------------------------------
    if verbose and n_significant > 0:
        sig_idx = np.where(significant)[0]
        order = sig_idx[np.argsort(p_values[sig_idx])]

        print(f"\n  Significant cells (sorted by p-value):")
        print(f"  {'cell':<10} {'α₁':>10} {'SE':>10} {'z':>8} {'p-value':>12}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

        for c in order:
            a, b = cells[c]
            print(
                f"  ({a:2d},{b:2d})   "
                f"{alpha1[c]:>+10.4f} {se1[c]:>10.4f} "
                f"{z_stats[c]:>+8.2f} {p_values[c]:>12.4f}"
            )

    # ------------------------------------------------------------------
    # Step 5: Diagnostics on U and V
    # ------------------------------------------------------------------
    if verbose:
        U_diag = np.diag(U)
        U_off = U - np.diag(U_diag)
        print("\nDiagnostics:")
        print(f"  U diag range: [{U_diag.min():.6f}, {U_diag.max():.6f}]")
        print(f"  max |U offdiag|: {np.max(np.abs(U_off)):.6f}")
        print(f"  V_diag mean: {np.mean(V_diag):.6f}, "
              f"range: [{np.min(V_diag):.6f}, {np.max(V_diag):.6f}]")

    return {
        "model": model,
        "alpha": alpha,
        "U": U,
        "V_diag": V_diag,
        "se": se,
        "z_stats": z_stats,
        "p_values": p_values,
        "significant": significant,
        "n_significant": n_significant,
        "cells": cells,
        "edges_per_cell": edges_per_cell,
        "disease_idx": disease_idx,
    }