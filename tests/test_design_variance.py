# tests/test_design_and_variance.py
from __future__ import annotations

import numpy as np

from src.design.fixed_effects import build_X_covars_diagnosis
from src.model.variance_structure import DiagonalVariance


def test_build_X_covars_diagnosis_basic():
    y = np.array([0, 1, 0, 1], dtype=int)
    X_cov, col_index = build_X_covars_diagnosis(y, add_intercept=True)

    # Shape: N x 2
    assert X_cov.shape == (4, 2)

    # Intercept is all ones
    intercept_col = X_cov[:, col_index["intercept"]]
    assert np.allclose(intercept_col, 1.0)

    # Diagnosis matches y
    diag_col = X_cov[:, col_index["diagnosis"]]
    assert np.allclose(diag_col, y.astype(float))


def test_diagonal_variance_from_scalar():
    E = 5
    sigma2 = 2.5
    V = DiagonalVariance.from_scalar(sigma2, E)

    assert V.E == E
    v = V.as_vector()
    assert v.shape == (E,)
    assert np.allclose(v, sigma2)


