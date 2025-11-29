from __future__ import annotations
import pytest
import numpy as np
import scipy.io as sio

from pathlib import Path

from src.io.io_cobre import load_cobre



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


