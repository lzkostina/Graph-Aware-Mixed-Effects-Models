from __future__ import annotations
import numpy as np

def power235_groups():
    """
    Return the authors' 13-community assignment after exclusions.
    The groups are 1-based ROI indices from the Power-264 atlas,
    with the 28 'Uncertain' nodes removed and node #75 missing,
    leaving exactly 235 ROIs split across 13 systems.

    Returns
    -------
    kept_rois : np.ndarray, shape (235,)
        1-based ROI indices kept in the analysis, concatenated in group order.
    sys_labels : np.ndarray, shape (235,)
        Community label in {1..13} for each kept ROI, aligned to kept_rois.
    per_group_indices : list[list[int]]
        The 13 lists of 1-based ROI indices per community (for reference).
    """
    groups = [
        list(range(13, 42)) + [255],                                  # 1
        list(range(42, 47)),                                           # 2
        list(range(47, 61)),                                           # 3
        list(range(61, 74)),                                           # 4
        [74] + list(range(76, 84)) + list(range(86, 132)) + [137, 139],# 5
        list(range(133, 137)) + [221],                                 # 6
        list(range(143, 174)),                                         # 7
        list(range(174, 182)) + list(range(186, 203)),                 # 8
        list(range(203, 221)),                                         # 9
        list(range(222, 235)),                                         # 10
        [138] + list(range(235, 243)),                                 # 11
        [251, 252] + list(range(256, 265)),                            # 12
        list(range(243, 247)),                                         # 13
    ]
    # sanity: total should be 235, no overlaps
    flat = [idx for g in groups for idx in g]
    assert len(flat) == 235, f"Expected 235 kept ROIs, got {len(flat)}"
    assert len(set(flat)) == 235, "Group lists overlap or contain duplicates"

    kept_rois = np.array(flat, dtype=int)                 # 1-based ROI ids
    sys_labels = np.concatenate([
        np.full(len(g), k + 1, dtype=int) for k, g in enumerate(groups)
    ])                                                    # 1..13 per kept ROI
    return kept_rois, sys_labels, groups


def make_masks_for_power235(n_full: int = 263, missing_node: int = 75):
    """
    Build masks to drop the 28 'Uncertain' ROIs (and #75 already missing) from an X
    whose columns correspond to the upper-tri edges on ROI IDs {1..264} excluding 75 .

    Parameters
    ----------
    n_full : int
        Number of ROIs used to build X (263 for 264 minus node 75).
    missing_node : int
        The single ROI ID excluded when forming X (75 in COBRE).

    Returns
    -------
    roi_ids_263 : (263,) int
        The 1-based ROI IDs underlying X's 263 nodes, in ascending order (all except 75).
    roi_keep_mask_263 : (263,) bool
        True for ROIs to keep (the 235 in the paper), False for Uncertain to drop.
    edge_keep_mask : (E,) bool
        True for edges whose BOTH endpoints are kept (use to slice columns of X).
    kept_roi_ids_235 : (235,) int
        1-based ROI IDs retained (should match power235_groups() up to ordering).
    sys_labels_235 : (235,) int
        Community labels {1..13} aligned to kept_roi_ids_235 (ascending ROI order).
    """

    # ROI IDs present in X (1..264 excluding 75), in ascending order:
    all_ids = np.arange(1, 265, dtype=int)
    roi_ids_263 = all_ids[all_ids != missing_node]  # (263,)

    # Paper's kept set (235 IDs)
    kept_rois, sys_labels_paper_order, _ = power235_groups()
    kept_set = set(kept_rois.tolist())

    # Which of the 263 are kept?
    roi_keep_mask_263 = np.array([rid in kept_set for rid in roi_ids_263], dtype=bool)
    kept_roi_ids_235 = roi_ids_263[roi_keep_mask_263]  # ascending by ROI id
    assert kept_roi_ids_235.size == 235, "Expected 235 kept ROIs after masking"

    # Build upper-tri indices for the 263-node graph
    n = roi_ids_263.size                       # 263
    iu_i, iu_j = np.triu_indices(n, k=1)       # E = n*(n-1)/2 = 34,453
    roi_keep = roi_keep_mask_263
    # Keep edges only if BOTH endpoints correspond to kept ROIs
    edge_keep_mask = roi_keep[iu_i] & roi_keep[iu_j]

    # Map each kept ROI ID to its community label (1..13)
    # First, build a lookup from ROI id to its community (from the paper groups)
    rid_to_comm = {rid: comm for rid, comm in zip(kept_rois.tolist(),
                                                  sys_labels_paper_order.tolist())}
    sys_labels_235 = np.array([rid_to_comm[rid] for rid in kept_roi_ids_235], dtype=int)

    return roi_ids_263, roi_keep_mask_263, edge_keep_mask, kept_roi_ids_235, sys_labels_235
