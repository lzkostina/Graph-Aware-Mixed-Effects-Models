from __future__ import annotations
import pytest
import numpy as np
import scipy.io as sio

from pathlib import Path
from src.io.io_cobre import load_cobre
from src.io.power_groups import power235_groups, make_masks_for_power235

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


