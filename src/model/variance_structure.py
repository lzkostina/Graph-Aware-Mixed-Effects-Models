from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class DiagonalVariance:
    """
    Diagonal residual covariance structure for V.

    We never build V as a full matrix; we just store its diagonal.
    """
    v_diag: np.ndarray  # shape (E,)

    @property
    def E(self) -> int:
        return self.v_diag.size

    def as_vector(self) -> np.ndarray:
        """Return the diagonal as a 1D array (view)."""
        return self.v_diag

    @classmethod
    def from_scalar(cls, sigma2: float, E: int) -> "DiagonalVariance":
        """Create sigma^2 * I_E."""
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")
        return cls(v_diag=np.full(E, float(sigma2), dtype=float))


@dataclass
class BlockDiagonalVariance:
    """
    Block-diagonal residual covariance structure for V, constant within each cell.

    Parameterization:
        - v_cell[c]    : variance for cell c
        - cell_id_of_edge[e] : which cell edge e belongs to

    We still represent V by its diagonal, but we enforce that all edges in a cell
    share the same variance parameter.

    This is convenient for EM updates: we can update v_cell[c] by pooling
    residuals over edges in cell c.
    """
    v_cell: np.ndarray          # shape (C,)
    cell_id_of_edge: np.ndarray # shape (E,)
    cell_edge_indices: List[np.ndarray]  # length C, indices of edges per cell

    @property
    def C(self) -> int:
        return self.v_cell.size

    @property
    def E(self) -> int:
        return self.cell_id_of_edge.size

    def as_vector(self) -> np.ndarray:
        """
        Return the diagonal of V as a length-E vector, where V[e,e] = v_cell[cell(e)].
        """
        return self.v_cell[self.cell_id_of_edge]

    @classmethod
    def from_scalar_per_cell(
        cls,
        sigma2_init: float,
        cell_id_of_edge: np.ndarray,
        C: int,
    ) -> "BlockDiagonalVariance":
        """
        Initialize with the same variance for every cell: v_cell[c] = sigma2_init.

        Parameters
        ----------
        sigma2_init : float
            Initial variance for each cell (must be >0).
        cell_id_of_edge : np.ndarray, shape (E,)
            For each edge, the index of its cell in [0..C-1].
        C : int
            Number of cells.

        Returns
        -------
        BlockDiagonalVariance
        """
        if sigma2_init <= 0:
            raise ValueError("sigma2_init must be positive.")

        cell_id_of_edge = np.asarray(cell_id_of_edge, dtype=int)
        if cell_id_of_edge.min() < 0 or cell_id_of_edge.max() >= C:
            raise ValueError("cell_id_of_edge must be in [0, C-1].")

        # Precompute which edges belong to which cell: a list of index arrays
        cell_edge_indices: List[np.ndarray] = []
        for c in range(C):
            inds = np.where(cell_id_of_edge == c)[0]
            cell_edge_indices.append(inds)

        v_cell = np.full(C, float(sigma2_init), dtype=float)
        return cls(v_cell=v_cell, cell_id_of_edge=cell_id_of_edge, cell_edge_indices=cell_edge_indices)
