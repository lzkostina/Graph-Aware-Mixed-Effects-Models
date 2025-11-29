from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy import io as sio
from typing import Tuple, Optional, Dict

def load_cobre(
    mat_dir: str | Path = "data/raw",
    x_key: str = "X",
    y_key: str = "Y",
    remap_labels: Optional[Dict[int, int]] = {-1: 0, 1: 1},
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load COBRE X and Y from MATLAB .mat files.

    Parameters
    ----------
    mat_dir : str | Path
        Directory containing X.mat and Y.mat.
    x_key : str
        Variable name for the design/edge matrix inside X.mat (default "X").
        Expected shape: (N, E).
    y_key : str
        Variable name for labels inside Y.mat (default "Y").
        Expected shapes: (N,), (N,1), or (1,N). Values should be in {-1, 1}.
    remap_labels : dict or None
        If provided, map original labels (e.g., {-1,1}) to new values (e.g., {0,1}).
        Set to None to keep labels as-is.

    Returns
    -------
    X : np.ndarray (N, E)
        Subjects × edges (Fisher-z correlations typically).
    y : np.ndarray (N,)
        Label per subject. If remap_labels is given, returned y is remapped (e.g., 0/1).
    healthy_idx : np.ndarray
        Integer indices of healthy controls (original label == -1).
    schizo_idx : np.ndarray
        Integer indices of schizophrenic patients (original label == 1).

    Raises
    ------
    FileNotFoundError
        If X.mat or Y.mat is missing.
    KeyError
        If x_key or y_key not found inside the respective .mat file.
    AssertionError
        On shape mismatches, non-finite values, or non-binary labels.
    """
    mat_dir = Path(mat_dir)
    x_path = mat_dir / "X.mat"
    y_path = mat_dir / "Y.mat"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    mats_X = sio.loadmat(x_path)
    mats_Y = sio.loadmat(y_path)

    if x_key not in mats_X:
        raise KeyError(f"Key '{x_key}' not found in {x_path.name}. Keys: {list(mats_X.keys())}")
    if y_key not in mats_Y:
        raise KeyError(f"Key '{y_key}' not found in {y_path.name}. Keys: {list(mats_Y.keys())}")

    X = np.asarray(mats_X[x_key])
    y_raw = np.asarray(mats_Y[y_key]).ravel().astype(int)

    # Basic shape checks
    assert X.ndim == 2, f"X must be 2D (subjects × edges); got shape {X.shape}"
    N, E = X.shape
    assert y_raw.shape[0] == N, f"Y length ({y_raw.shape[0]}) must match #rows of X ({N})"

    # Finite check
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise AssertionError(f"X contains NaN/Inf at positions like {bad[:5].tolist()} (showing up to 5).")

    # Label checks in original coding
    uniq = set(np.unique(y_raw).tolist())
    expected = {-1, 1}
    assert uniq.issubset(expected), f"Y must contain only {-1,1}; got {sorted(uniq)}"

    # Indices by original labels (useful even if we remap)
    schizo_idx = np.where(y_raw == 1)[0]
    healthy_idx = np.where(y_raw == -1)[0]

    # Optional remap to 0/1 (or anything you pass)
    if remap_labels is not None:
        try:
            y = np.vectorize(remap_labels.__getitem__)(y_raw)
        except KeyError as e:
            raise AssertionError(f"Label {e} not in remap_labels keys {list(remap_labels.keys())}.") from None
    else:
        y = y_raw.copy()

    # Logging
    print("Loaded COBRE:")
    print(f"  X: {X.shape} (subjects × edges)")
    print(f"  y: {y.shape} (labels; unique={sorted(set(y.tolist()))})")
    print(f"  healthy n={len(healthy_idx)}, schizo n={len(schizo_idx)}")

    return X, y, healthy_idx, schizo_idx

