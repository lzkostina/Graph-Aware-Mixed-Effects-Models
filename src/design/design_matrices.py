from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Tuple
from src.design.cells import build_cells


def build_Z_from_cell_ids(cell_id_of_edge: np.ndarray, C: int) -> sp.csr_matrix:
    """
    Build Z (E x C): Z[k, c] = 1 if edge k belongs to cell c, else 0.
    """
    E = cell_id_of_edge.size
    rows = np.arange(E)
    cols = cell_id_of_edge.astype(int)
    data = np.ones(E, dtype=int)
    return sp.csr_matrix((data, (rows, cols)), shape=(E, C))


def build_Z_for_X235(
    roi_ids_263: np.ndarray,
    roi_keep_mask_263: np.ndarray,
    edge_keep_mask: np.ndarray,
    sys_labels_235: np.ndarray,
    K: int = 13,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Construct Z (E x C) for the *kept* edge columns of X (i.e., X_235 = X[:, edge_keep_mask]).

    Parameters
    ----------
    roi_ids_263 : (263,) int
        Ascending 1-based ROI IDs underlying the original 263-node graph (all except 75).
    roi_keep_mask_263 : (263,) bool
        True for ROIs to keep (the 235 used in the paper), False for 'Uncertain' (drop).
    edge_keep_mask : (E263,) bool
        True for edges whose BOTH endpoints are kept (use to slice columns of X).
    sys_labels_235 : (235,) int
        Community labels {1..13} for the *kept* ROIs, aligned to kept ROI IDs in ascending order.
    K : int
        Number of communities (13 in the paper).

    Returns
    -------
    Z : csr_matrix, shape (E235, C) where C = K*(K+1)//2
        One-hot design: Z[k, c] = 1 if edge k belongs to cell c = (a,b) with a<=b.
        Edge row ordering matches the kept columns of X, i.e., X_235 = X[:, edge_keep_mask].
    cells : (C, 2) int
        The (a,b) pairs (1-based) in lexicographic order with a<=b.
    cell_id_of_edge : (E235,) int
        For each kept edge column, the index of its cell in [0..C-1].
    """
    # sanity checks
    roi_ids_263 = np.asarray(roi_ids_263, int)
    roi_keep_mask_263 = np.asarray(roi_keep_mask_263, bool)
    edge_keep_mask = np.asarray(edge_keep_mask, bool)
    sys_labels_235 = np.asarray(sys_labels_235, int)
    assert roi_ids_263.shape == roi_keep_mask_263.shape
    n263 = roi_ids_263.size
    iu_i_263, iu_j_263 = np.triu_indices(n263, k=1)
    assert edge_keep_mask.shape == (iu_i_263.size,), "edge_keep_mask length must match edges of n=263"

    # map 263-node indices -> 235-node indices (-1 for dropped)
    kept_idx_263 = np.where(roi_keep_mask_263)[0]                 # positions in [0..262] that are kept
    # kept ROI IDs in ascending order must match those used for sys_labels_235
    kept_roi_ids_235 = roi_ids_263[kept_idx_263]
    # build reverse map: for each 263 index, where does it land in 235 (or -1 if dropped)
    map263_to_235 = np.full(n263, -1, dtype=int)
    map263_to_235[kept_idx_263] = np.arange(kept_idx_263.size)    # 0..234

    # endpoints of kept edges (in 263 indexing), then map to 235 indexing
    i263_all, j263_all = iu_i_263[edge_keep_mask], iu_j_263[edge_keep_mask]
    i235 = map263_to_235[i263_all]
    j235 = map263_to_235[j263_all]
    # get community labels for endpoints
    a = sys_labels_235[i235]
    b = sys_labels_235[j235]
    # ensure unordered cell (a<=b)
    a2 = np.minimum(a, b)
    b2 = np.maximum(a, b)

    # cells list and mapping (a,b)->cell index
    cells = build_cells(K, base=1)                                # (C,2) with a,b in 1..13
    C = cells.shape[0]
    # linear map: cell_idx = dict[(a,b)] -> 0..C-1
    cell_to_idx = { (cells[r,0], cells[r,1]): r for r in range(C) }
    # vectorized mapping: turn (a2,b2) pairs into indices
    # fast way: compute ranks by searching in a flat key space
    # but since K is tiny (13), a loop is fine:
    cell_id_of_edge = np.empty_like(a2, dtype=int)
    for k in range(a2.size):
        cell_id_of_edge[k] = cell_to_idx[(int(a2[k]), int(b2[k]))]

    # build Z
    E235 = a2.size
    rows = np.arange(E235, dtype=int)
    cols = cell_id_of_edge
    data = np.ones(E235, dtype=int)
    Z = sp.csr_matrix((data, (rows, cols)), shape=(E235, C))
    return Z, cells, cell_id_of_edge

