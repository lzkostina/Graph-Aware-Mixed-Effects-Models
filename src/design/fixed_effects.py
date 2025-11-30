from __future__ import annotations

from typing import Tuple
import numpy as np


def build_X_covars_diagnosis(
    y: np.ndarray,
    add_intercept: bool = True,
) -> Tuple[np.ndarray, dict[str, int]]:
    """
    Build a simple fixed-effects design matrix for COBRE:
    intercept + diagnosis (0/1).

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Subject-level diagnosis labels, typically 0 for control and 1 for patient.
    add_intercept : bool, default True
        Whether to include an intercept column of ones.

    Returns
    -------
    X_cov : np.ndarray, shape (N, p)
        Fixed-effects design matrix. By default:
            column 0: intercept
            column 1: diagnosis
    col_index : dict[str, int]
        Mapping from semantic name to column index, e.g.
            {"intercept": 0, "diagnosis": 1}
        (If add_intercept=False, "diagnosis" will be 0.)
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {y.shape}")

    uniq = np.unique(y)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"y is expected to contain only 0/1 labels; got {uniq}")

    N = y.shape[0]
    cols = []
    col_index: dict[str, int] = {}

    if add_intercept:
        cols.append(np.ones(N, dtype=float))
        col_index["intercept"] = 0
        col_index["diagnosis"] = 1
    else:
        col_index["diagnosis"] = 0

    cols.append(y.astype(float))

    X_cov = np.column_stack(cols)
    return X_cov, col_index
