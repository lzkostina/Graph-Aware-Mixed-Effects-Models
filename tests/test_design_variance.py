# tests/test_design_and_variance.py
from __future__ import annotations

import numpy as np

from src.design.fixed_effects import build_X_covars_diagnosis
from src.model.variance_structure import DiagonalVariance, BlockDiagonalVariance


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


def test_block_diagonal_variance_from_scalar_per_cell():
    # 5 edges, belonging to cells [0, 1, 1, 0, 2]
    cell_id_of_edge = np.array([0, 1, 1, 0, 2], dtype=int)
    C = 3
    sigma2_init = 1.5

    Vb = BlockDiagonalVariance.from_scalar_per_cell(
        sigma2_init=sigma2_init, cell_id_of_edge=cell_id_of_edge, C=C
    )

    # Basic properties
    assert Vb.C == C
    assert Vb.E == cell_id_of_edge.size
    assert Vb.v_cell.shape == (C,)
    assert np.allclose(Vb.v_cell, sigma2_init)

    # Diagonal representation: each edge gets the variance of its cell
    v_diag = Vb.as_vector()
    expected = np.array(
        [sigma2_init, sigma2_init, sigma2_init, sigma2_init, sigma2_init], dtype=float
    )
    assert np.allclose(v_diag, expected)

    # Cell edge indices cover all edges without duplication
    all_inds = np.concatenate(Vb.cell_edge_indices)
    all_inds_sorted = np.sort(all_inds)
    assert np.array_equal(all_inds_sorted, np.arange(Vb.E))
