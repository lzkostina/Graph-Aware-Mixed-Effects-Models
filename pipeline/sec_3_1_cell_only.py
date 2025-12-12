"""
Section 3.1 with CORRECT GLS Standard Errors (cell-only model)

We use the exact GLS variance formula from Kim, Kessler & Levina (2023):

    Var(vec(α̂)) = (Σ_m X_m^T Σ^{-1} X_m)^(-1)

For the *cell-only* model, we can show:

    Σ_m X_m^T Σ^{-1} X_m = (X^T X) ⊗ W,

where W is a C × C matrix depending on U and V via:

    w[c] = Σ_{e∈cell c} 1 / v_e
    M    = U^{-1} + diag(w)
    W    = diag(w) - M^{-1} ⊙ (w w^T)

This gives *proper GLS standard errors* for cell-level fixed effects α,
while the model fit itself is the cell-only EM (no edge-level η).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats

from src.model.graph_aware_em import GraphAwareEM_CellOnly, EMConfig


# =============================================================================
# PROPER GLS STANDARD ERRORS (cell-only model)
# =============================================================================

def compute_gls_variance_proper(
    X: np.ndarray,
    edges_per_cell: List[np.ndarray],
    U: np.ndarray,
    V_diag: np.ndarray,
) -> np.ndarray:
    """
    Compute the PROPER GLS variance matrix for cell-level effects α
    under the cell-only model.

    For the cell-only model, the information matrix has the form:

        I = Σ_m X_m^T Σ^{-1} X_m = (X^T X) ⊗ W,

    where W is defined via:

        w[c] = Σ_{e∈cell c} 1 / v_e
        M    = U^{-1} + diag(w)
        W    = diag(w) - M^{-1} ⊙ (w w^T)

    Parameters
    ----------
    X : (N, p) array
        Covariate matrix.
    edges_per_cell : list of arrays
        Each entry is a 1D array of edge indices in that cell.
    U : (C, C) array
        Random effects covariance (from EM).
    V_diag : (E,) array
        Residual variances at the edge level.

    Returns
    -------
    var_alpha : (C, p, p) array
        Variance matrices for each cell's coefficient vector α^(c).
    """
    N, p = X.shape
    C = len(edges_per_cell)

    # 1. w[c] = sum_{e in cell c} 1 / v_e
    w = np.zeros(C, dtype=float)
    for c, edge_idx in enumerate(edges_per_cell):
        if edge_idx.size == 0:
            w[c] = 0.0
        else:
            w[c] = np.sum(1.0 / V_diag[edge_idx])

    # 2. M = U^{-1} + diag(w)  (regularize if needed)
    try:
        U_inv = np.linalg.inv(U)
    except np.linalg.LinAlgError:
        U_inv = np.linalg.inv(U + 1e-8 * np.eye(C))

    M = U_inv + np.diag(w)

    # 3. M^{-1}
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)

    # 4. W = diag(w) - M^{-1} ⊙ (w w^T)
    wwT = np.outer(w, w)
    W = np.diag(w) - M_inv * wwT   # elementwise product

    # 5. Information matrix: I = (X^T X) ⊗ W
    XtX = X.T @ X  # (p, p)
    info_matrix = np.kron(W, XtX)  # (C*p, C*p)

    # 6. Invert to get Var(vec(α̂))
    try:
        var_vec = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        var_vec = np.linalg.pinv(info_matrix)

    # 7. Extract per-cell blocks
    var_alpha = np.zeros((C, p, p), dtype=float)
    for c in range(C):
        start = c * p
        stop = (c + 1) * p
        var_alpha[c] = var_vec[start:stop, start:stop]

    return var_alpha


def compute_gls_se_proper(
    X: np.ndarray,
    edges_per_cell: List[np.ndarray],
    U: np.ndarray,
    V_diag: np.ndarray,
) -> np.ndarray:
    """
    Proper GLS standard errors for cell-level fixed effects α.

    Parameters
    ----------
    X : (N, p)
    edges_per_cell : list of 1D index arrays
    U : (C, C)
    V_diag : (E,)

    Returns
    -------
    se : (C, p) array
        Standard errors per cell and covariate.
    """
    var_alpha = compute_gls_variance_proper(X, edges_per_cell, U, V_diag)
    C, p, _ = var_alpha.shape

    se = np.zeros((C, p), dtype=float)
    for c in range(C):
        diag_c = np.diag(var_alpha[c])
        # numerical guard
        diag_c = np.maximum(diag_c, 0.0)
        se[c] = np.sqrt(diag_c)

    return se


def compute_gls_se_simplified(
    X: np.ndarray,
    edges_per_cell: List[np.ndarray],
    U: np.ndarray,
    V_diag: np.ndarray,
) -> np.ndarray:
    """
    Simplified GLS SE (treats cells as independent).

    Only valid when:
      1. U is diagonal (or nearly so)
      2. Cross-cell correlations are negligible.
    """
    N, p = X.shape
    C = len(edges_per_cell)

    XtX_inv = np.linalg.inv(X.T @ X)

    se = np.zeros((C, p), dtype=float)
    for c, edge_idx in enumerate(edges_per_cell):
        n_c = len(edge_idx)
        if n_c == 0:
            se[c] = np.nan
            continue

        tau2_c = U[c, c]
        sigma2_c = float(np.mean(V_diag[edge_idx]))
        var_cell_mean = tau2_c + sigma2_c / n_c
        se[c] = np.sqrt(np.diag(XtX_inv) * var_cell_mean)

    return se


def compare_se_formulas(
    X: np.ndarray,
    edges_per_cell: List[np.ndarray],
    U: np.ndarray,
    V_diag: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Compare simplified vs proper GLS SE formulas.

    Returns a dictionary with both SE arrays and their elementwise ratio.
    """
    se_simplified = compute_gls_se_simplified(X, edges_per_cell, U, V_diag)
    se_proper = compute_gls_se_proper(X, edges_per_cell, U, V_diag)

    ratio = se_simplified / se_proper

    return {
        "se_simplified": se_simplified,
        "se_proper": se_proper,
        "ratio": ratio,
        "ratio_mean": float(np.nanmean(ratio)),
        "ratio_std": float(np.nanstd(ratio)),
        "ratio_min": float(np.nanmin(ratio)),
        "ratio_max": float(np.nanmax(ratio)),
    }


# =============================================================================
# HYPOTHESIS TESTING WITH PROPER GLS SE
# =============================================================================

def test_cell_effects(
    alpha: np.ndarray,
    se: np.ndarray,
    significance_level: float = 0.05,
) -> Dict[str, np.ndarray | float]:
    """
    Test H0: α₁^{(c)} = 0 for each cell (disease effect).

    Parameters
    ----------
    alpha : (C, p) array
        Estimated cell effects.
    se : (C, p) array
        Standard errors (e.g., from compute_gls_se_proper).
    significance_level : float
        Significance level α.

    Returns
    -------
    results : dict
        Contains alpha1, se1, z_stats, p_values, significant, n_significant.
    """
    C, p = alpha.shape
    if p < 2:
        raise ValueError("Expected at least 2 covariates, with disease effect in column 1.")

    alpha1 = alpha[:, 1]
    se1 = se[:, 1]

    z_stats = alpha1 / se1
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))
    significant = p_values < significance_level

    return {
        "alpha1": alpha1,
        "se1": se1,
        "z_stats": z_stats,
        "p_values": p_values,
        "significant": significant,
        "n_significant": int(np.sum(significant)),
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION (Section 3.1, cell-only)
# =============================================================================

def run_section31_corrected(
    X_235: np.ndarray,
    X_cov: np.ndarray,
    cells: np.ndarray,
    cell_id_of_edge: np.ndarray,
    fit_max_iter: int = 100,
    significance_level: float = 0.05,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run Section 3.1 analysis with PROPER GLS standard errors
    under the cell-only EM model (no edge-level η).

    Parameters
    ----------
    X_235 : (N, E) array
        Edge weights for 235-ROI network (after masking).
    X_cov : (N, p) array
        Covariate matrix (e.g. [1, diagnosis]).
    cells : (C, 2) array
        Cell indices (system_a, system_b).
    cell_id_of_edge : (E,) array
        Cell index for each edge.
    fit_max_iter : int
        Max EM iterations.
    significance_level : float
        α for hypothesis tests.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with alpha, U, V_diag, SEs, test results, etc.
    """
    if verbose:
        print("=" * 60)
        print("Section 3.1 Analysis with PROPER GLS Standard Errors")
        print("=" * 60)

    N, E = X_235.shape
    C = len(cells)

    # Build edges_per_cell as a list of edge index arrays
    edges_per_cell = [np.where(cell_id_of_edge == c)[0] for c in range(C)]

    if verbose:
        print(f"\nData: N={N}, E={E}, C={C}")

    # Step 1: Fit cell-only EM model
    if verbose:
        print("\nStep 1: Fitting cell-only model via EM...")

    config = EMConfig(max_iter=fit_max_iter, tol=1e-4, verbose=verbose)
    model = GraphAwareEM_CellOnly(
        Y=X_235,
        X=X_cov,
        cell_id_of_edge=cell_id_of_edge,
        edges_per_cell=edges_per_cell,
        config=config,
    )
    model.fit()

    alpha = model.get_cell_effects()
    U = model.get_random_effects_covariance()
    V_diag = model.get_residual_variances()

    # Step 2: Compare SE formulas
    if verbose:
        print("\nStep 2: Computing standard errors...")
        print("  Comparing simplified vs proper GLS formulas...")

    comparison = compare_se_formulas(X_cov, edges_per_cell, U, V_diag)

    if verbose:
        print("\n  SE ratio (simplified / proper):")
        print(f"    Mean:  {comparison['ratio_mean']:.3f}")
        print(f"    Std:   {comparison['ratio_std']:.3f}")
        print(f"    Range: [{comparison['ratio_min']:.3f}, {comparison['ratio_max']:.3f}]")

        if comparison["ratio_mean"] > 1.1:
            print("\n  ⚠ Simplified formula OVERESTIMATES SE (more conservative)")
        elif comparison["ratio_mean"] < 0.9:
            print("\n  ⚠ Simplified formula UNDERESTIMATES SE (anti-conservative)")
        else:
            print("\n  ✓ Formulas are reasonably close")

    # Step 3: Hypothesis testing
    if verbose:
        print("\nStep 3: Hypothesis testing...")

    results_simplified = test_cell_effects(alpha, comparison["se_simplified"], significance_level)
    results_proper = test_cell_effects(alpha, comparison["se_proper"], significance_level)

    if verbose:
        print(f"\n  Using SIMPLIFIED SE: {results_simplified['n_significant']} significant cells")
        print(f"  Using PROPER GLS SE: {results_proper['n_significant']} significant cells")
        print("\n  Paper reports: 13 significant cells")

    # Step 4: Some diagnostics on U
    if verbose:
        print("\nDiagnostics:")
        U_diag = np.diag(U)
        U_offdiag = U - np.diag(U_diag)
        print(f"  U diagonal range: [{U_diag.min():.4f}, {U_diag.max():.4f}]")
        print(f"  U off-diagonal |max|: {np.max(np.abs(U_offdiag)):.4f}")

        if np.max(np.abs(U_offdiag)) > 0.01 * np.mean(U_diag):
            print("  → U has significant off-diagonal terms (proper GLS needed)")
        else:
            print("  → U is nearly diagonal (simplified formula may be OK)")

    return {
        "alpha": alpha,
        "U": U,
        "V_diag": V_diag,
        "se_simplified": comparison["se_simplified"],
        "se_proper": comparison["se_proper"],
        "se_ratio": comparison["ratio"],
        "results_simplified": results_simplified,
        "results_proper": results_proper,
        "cells": cells,
        "edges_per_cell": edges_per_cell,
    }
