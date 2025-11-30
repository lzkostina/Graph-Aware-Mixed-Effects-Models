from __future__ import annotations
import pytest
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from pathlib import Path
from src.io.io_cobre import load_cobre
from src.io.power_groups import power235_groups, make_masks_for_power235
from src.design.design_matrices import build_Z_from_cell_ids, build_Z_for_X235

def test_load_cobre_roundtrip(tmp_path: Path):
    """
    Create small fake X.mat and Y.mat, then check that load_cobre:
    - loads shapes correctly
    - remaps labels as expected
    - returns correct healthy / schizo indices.
    """
    mat_dir = tmp_path

    # Fake data: 4 subjects, 6 edges
    X = np.arange(24).reshape(4, 6).astype(float)
    y = np.array([-1, 1, -1, 1], dtype=int)  # original labels

    sio.savemat(mat_dir / "X.mat", {"X": X})
    sio.savemat(mat_dir / "Y.mat", {"Y": y})

    X_loaded, y_loaded, healthy_idx, schizo_idx = load_cobre(mat_dir=mat_dir)

    # Shapes
    assert X_loaded.shape == (4, 6)
    assert y_loaded.shape == (4,)

    # Value equality
    np.testing.assert_array_equal(X_loaded, X)

    # Default remap {-1: 0, 1: 1}
    assert set(y_loaded.tolist()) == {0, 1}
    np.testing.assert_array_equal(healthy_idx, np.array([0, 2]))
    np.testing.assert_array_equal(schizo_idx, np.array([1, 3]))


def test_load_cobre_roundtrip(tmp_path: Path):
    """
    Create small fake X.mat and Y.mat, then check that load_cobre:
    - loads shapes correctly
    - remaps labels as expected
    - returns correct healthy / schizo indices.
    """
    mat_dir = tmp_path

    # Fake data: 4 subjects, 6 edges
    X = np.arange(24).reshape(4, 6).astype(float)
    y = np.array([-1, 1, -1, 1], dtype=int)  # original labels

    sio.savemat(mat_dir / "X.mat", {"X": X})
    sio.savemat(mat_dir / "Y.mat", {"Y": y})

    X_loaded, y_loaded, healthy_idx, schizo_idx = load_cobre(mat_dir=mat_dir)

    # Shapes
    assert X_loaded.shape == (4, 6)
    assert y_loaded.shape == (4,)

    # Value equality
    np.testing.assert_array_equal(X_loaded, X)

    # Default remap {-1: 0, 1: 1}
    assert set(y_loaded.tolist()) == {0, 1}
    np.testing.assert_array_equal(healthy_idx, np.array([0, 2]))
    np.testing.assert_array_equal(schizo_idx, np.array([1, 3]))


def test_power235_groups_basic_properties():
    """
    Check that power235_groups returns:
    - exactly 235 ROI IDs
    - no duplicates
    - sys_labels aligned and in 1..13.
    """
    kept_rois, sys_labels, groups = power235_groups()

    assert kept_rois.shape == (235,)
    assert sys_labels.shape == (235,)

    # No duplicates
    assert len(set(kept_rois.tolist())) == 235

    # Community labels in {1..13}
    assert sys_labels.min() >= 1
    assert sys_labels.max() <= 13

    # Groups structure consistent
    assert len(groups) == 13
    flat_again = [idx for g in groups for idx in g]
    assert len(flat_again) == 235
    assert len(set(flat_again)) == 235

def test_build_Z_from_cell_ids_one_hot():
    """
    Simple test: for a small cell_id_of_edge vector, Z should be one-hot with
    exactly one '1' per row and correct column positions.
    """
    cell_id_of_edge = np.array([0, 1, 1, 2, 0], dtype=int)  # E = 5, C = 3
    C = 3
    Z = build_Z_from_cell_ids(cell_id_of_edge, C)

    assert isinstance(Z, sp.csr_matrix)
    assert Z.shape == (5, 3)

    Z_dense = Z.toarray()
    # One '1' per row
    assert (Z_dense.sum(axis=1) == 1).all()

    # Column locations match cell_id_of_edge
    expected = np.zeros_like(Z_dense)
    for e, c in enumerate(cell_id_of_edge):
        expected[e, c] = 1
    np.testing.assert_array_equal(Z_dense, expected)


def test_build_Z_for_X235_basic_properties():
    """
    End-to-end sanity test that build_Z_for_X235:
    - produces Z with one '1' per row,
    - has shape (E235, C) where C = K*(K+1)//2,
    - cell_id_of_edge is in [0..C-1].
    """
    # Use the real masks
    roi_ids_263, roi_keep_mask_263, edge_keep_mask, kept_roi_ids_235, sys_labels_235 = make_masks_for_power235()

    K = 13
    Z, cells, cell_id_of_edge = build_Z_for_X235(
        roi_ids_263=roi_ids_263,
        roi_keep_mask_263=roi_keep_mask_263,
        edge_keep_mask=edge_keep_mask,
        sys_labels_235=sys_labels_235,
        K=K,
    )

    C = K * (K + 1) // 2
    assert cells.shape == (C, 2)
    assert Z.shape[1] == C
    assert cell_id_of_edge.shape[0] == Z.shape[0]

    # Z is one-hot per row
    Z_dense_row_sums = np.asarray(Z.sum(axis=1)).ravel()
    assert np.all(Z_dense_row_sums == 1), "Each edge should belong to exactly one cell"

    # cell_id_of_edge range
    assert cell_id_of_edge.min() >= 0
    assert cell_id_of_edge.max() < C
