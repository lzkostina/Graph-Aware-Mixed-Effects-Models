from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

def build_cells(K: int, base: int = 1) -> np.ndarray:
    """
    Return all unordered community-cell pairs (a, b) with a <= b.

    Parameters
    ----------
    K : int
        Number of communities/systems.
    base : int
        1 for MATLAB-style labels {1..K}, 0 for Python-style {0..K-1}.

    Returns
    -------
    nC : np.ndarray, shape (K*(K+1)//2, 2), dtype=int
        Each row is a pair [a, b] with a <= b in lexicographic order:
        [base, base], [base, base+1], ..., [base, base+(K-1)],
        [base+1, base+1], ..., [base+(K-1), base+(K-1)].

    Raises
    ------
    AssertionError
        If K < 1 or base not in {0, 1}.
    """
    assert isinstance(K, int) and K >= 1, f"K must be a positive int, got {K}"
    assert base in (0, 1), f"base must be 0 or 1, got {base}"

    # Create grid of labels and keep upper-triangular (including diagonal)
    labels = np.arange(base, base + K, dtype=int)
    C1, C2 = np.meshgrid(labels, labels, indexing="ij")
    mask = C1 <= C2
    nC = np.column_stack((C1[mask], C2[mask]))
    return nC


def map_edges_to_cells(
    sys_labels: np.ndarray,
    base: int = 1,
):
    """
    Given ROI -> system labels, return:
      - cells: array of (a,b) with a<=b (shape Cx2)
      - tri_idx: (i_idx, j_idx) upper-tri indices (k=1)
      - cell_id_of_edge: array (E,) mapping each edge k to cell index in [0..C-1]
      - edges_by_cell: dict[int -> np.ndarray of (i,j) pairs] for quick access

    Parameters
    ----------
    sys_labels : array-like of shape (n,)
        Community label for each ROI. Either {1..K} (base=1) or {0..K-1} (base=0).
    base : int
        1 for labels in {1..K}, 0 for {0..K-1}.

    Returns
    -------
    cells : np.ndarray
    tri_idx : Tuple[np.ndarray, np.ndarray]
    cell_id_of_edge : np.ndarray
    edges_by_cell : Dict[int, np.ndarray]
    """
    sys_labels = np.asarray(sys_labels, dtype=int)
    n = sys_labels.size

    assert n >= 2, "need at least 2 ROIs"
    assert base in (0, 1), "base must be 0 or 1"

    if base == 1:
        assert (sys_labels.min()).item() == 1, "labels must start at 1 for base=1"
        K = (sys_labels.max()).item()
    else:
        assert (sys_labels.min()).item() == 0, "labels must start at 0 for base=0"
        K = (sys_labels.max()).item() + 1

    assert len(np.unique(sys_labels)) == K
    # list of all cells (a,b) with a<=b
    cells = build_cells(K, base=base)
    C = cells.shape[0]

    # upper-triangular edge indices
    iu_i, iu_j = np.triu_indices(n, k=1)
    E = iu_i.size

    # map (a,b) -> cell index
    cell_index: Dict[Tuple[int, int], int] = {
        (cells[r, 0], cells[r, 1]): r for r in range(C)
    }

    # fill edge -> cell mapping and buckets
    cell_id_of_edge = np.empty(E, dtype=int)
    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(C)]

    for k, (i, j) in enumerate(zip(iu_i, iu_j)):
        a = sys_labels[i]
        b = sys_labels[j]
        if a > b:
            a, b = b, a
        idx = cell_index[(a, b)]
        cell_id_of_edge[k] = idx
        buckets[idx].append((i, j))

    edges_by_cell = {c: np.array(pairs, dtype=int).reshape(-1, 2) if pairs else
                        np.empty((0, 2), dtype=int)
                     for c, pairs in enumerate(buckets)}

    return cells, (iu_i, iu_j), cell_id_of_edge, edges_by_cell