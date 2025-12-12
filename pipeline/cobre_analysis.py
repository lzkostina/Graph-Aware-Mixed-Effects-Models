"""
Complete Pipeline for COBRE Analysis

This module demonstrates the full workflow:
1. Load COBRE data
2. Preprocess (filter ROIs, build cell structure)
3. Fit the graph-aware mixed effects model via EM
4. Extract results

Based on Kim, Kessler, and Levina (2023).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np

from src.design.cells import build_cells, map_edges_to_cells
from src.io.power_groups import make_masks_for_power235  # <- make sure this exists
from src.model.graph_aware_em import (
    GraphAwareEM,
    GraphAwareEM_CellOnly,
    EMConfig,
)


class COBREAnalysis:
    """
    Complete analysis pipeline for COBRE schizophrenia data.

    This class handles:
    1. Data preprocessing (ROI filtering, cell structure)
    2. Model fitting via EM
    3. Results extraction and formatting
    """

    # System names from Table 1 of the paper.
    # NOTE: These are keyed by 1-based system indices.
    SYSTEM_NAMES: Dict[int, str] = {
        1: "Sensory/somatomotor Hand",
        2: "Sensory/somatomotor Mouth",
        3: "Cingulo-opercular Task Control",
        4: "Auditory",
        5: "Default mode",
        6: "Memory retrieval",
        7: "Visual",
        8: "Fronto-parietal Task Control",
        9: "Salience",
        10: "Subcortical",
        11: "Ventral attention",
        12: "Dorsal attention",
        13: "Cerebellar",
    }

    def __init__(self) -> None:
        """Initialize analysis (preprocessing is done lazily)."""
        # Cell structure
        self.cells: Optional[np.ndarray] = None         # shape (C, 2), system indices (1..13)
        self.cell_id_of_edge: Optional[np.ndarray] = None  # shape (E,), cell index for each edge
        self.edges_per_cell: Optional[List[np.ndarray]] = None  # list of arrays of edge indices
        self.sys_labels: Optional[np.ndarray] = None    # shape (235,), system label for each ROI
        self.edge_keep_mask: Optional[np.ndarray] = None  # shape (E_263,), bool mask for X_raw

        # Data
        self.Y: Optional[np.ndarray] = None   # (N, E) edge weights after filtering
        self.X: Optional[np.ndarray] = None   # (N, p) covariate matrix
        self.disease_labels: Optional[np.ndarray] = None  # (N,) 0=healthy, 1=schizo

        # Fitted model
        self.model: Optional[GraphAwareEM] = None

        self._preprocessed: bool = False

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    def preprocess(self) -> None:
        """
        Set up cell structure for 235 ROIs / 13 systems.

        This uses:
            - make_masks_for_power235() to restrict to 235 ROIs
            - map_edges_to_cells() to build cells and edge→cell mapping
        """
        if self._preprocessed:
            return

        (
            roi_ids_263,
            roi_keep_mask_263,
            edge_keep_mask,
            kept_roi_ids_235,
            sys_labels_235,
        ) = make_masks_for_power235()

        # Build cell structure.
        # NOTE: base=1 so that system indices a,b are in {1,...,13}, which matches SYSTEM_NAMES.
        cells, tri_idx, cell_id_of_edge, edges_by_cell = map_edges_to_cells(
            sys_labels_235,
            base=1,
        )
        C = cells.shape[0]
        E = cell_id_of_edge.size

        # Build edges_per_cell as a list of edge index arrays (0..E-1)
        edges_per_cell = [np.where(cell_id_of_edge == c)[0] for c in range(C)]

        # Optional sanity check (mirrors GraphAwareEM._validate_inputs logic)
        all_edges = np.concatenate(edges_per_cell)
        if len(all_edges) != E or set(all_edges) != set(range(E)):
            raise RuntimeError("Internal error: edges_per_cell does not partition [0, E-1].")

        self.cells = cells
        self.cell_id_of_edge = cell_id_of_edge
        self.edges_per_cell = edges_per_cell
        self.sys_labels = sys_labels_235
        self.edge_keep_mask = edge_keep_mask

        self._preprocessed = True

        print("Preprocessing complete:")
        print("  ROIs: 235")
        print("  Systems: 13")
        print(f"  Cells: {len(cells)}")
        print(f"  Edges: {len(cell_id_of_edge)}")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    def load_data(self, X_raw: np.ndarray, y_labels: np.ndarray) -> None:
        """
        Load edge weights and labels.

        Parameters
        ----------
        X_raw : (N, E_263) array
            Edge weights from COBRE, before filtering.
            E_263 = 263*262/2 = 34,453 edges.
        y_labels : (N,) array
            Disease labels: 0 = healthy, 1 = schizophrenia.
        """
        self.preprocess()

        N, E_263 = X_raw.shape
        if self.edge_keep_mask is None:
            raise RuntimeError("Preprocessing failed: edge_keep_mask is not set.")

        if E_263 != self.edge_keep_mask.size:
            raise ValueError(
                f"X_raw has {E_263} edges, but edge_keep_mask has "
                f"{self.edge_keep_mask.size} entries."
            )

        # Filter edges to keep only those between kept ROIs
        self.Y = X_raw[:, self.edge_keep_mask]

        # Build covariate matrix: [1, disease]
        self.X = np.column_stack([np.ones(N), y_labels])
        self.disease_labels = y_labels

        print("Data loaded:")
        print(f"  Subjects: {N} "
              f"({np.sum(y_labels == 0)} healthy, {np.sum(y_labels == 1)} schizo)")
        print(f"  Y shape: {self.Y.shape}")
        print(f"  X shape: {self.X.shape}")

    # -------------------------------------------------------------------------
    # Model fitting
    # -------------------------------------------------------------------------
    def fit(
        self,
        model_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ) -> "COBREAnalysis":
        """
        Fit the graph-aware mixed effects model.

        Parameters
        ----------
        model_type : {"full", "cell_only"}
            "full" for model with edge effects η
            "cell_only" for model without edge effects
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance for log-likelihood.
        verbose : bool
            Print progress.

        Returns
        -------
        self : COBREAnalysis
        """
        if self.Y is None or self.X is None or self.cell_id_of_edge is None:
            raise ValueError("Call load_data() before fit().")

        if self.edges_per_cell is None:
            raise RuntimeError("Preprocessing failed: edges_per_cell is not set.")

        config = EMConfig(
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

        if model_type == "full":
            model_cls = GraphAwareEM
        elif model_type == "cell_only":
            model_cls = GraphAwareEM_CellOnly
        else:
            raise ValueError(
                f"model_type must be 'full' or 'cell_only', got {model_type!r}"
            )

        self.model = model_cls(
            Y=self.Y,
            X=self.X,
            cell_id_of_edge=self.cell_id_of_edge,
            edges_per_cell=self.edges_per_cell,
            config=config,
        )

        self.model.fit()
        return self

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    def get_cell_effects(self) -> np.ndarray:
        """
        Get cell-level fixed effects α.

        Returns
        -------
        alpha : (C, p) array
            alpha[:, 0] = intercept (healthy mean)
            alpha[:, 1] = disease effect (schizo - healthy)
        """
        if self.model is None:
            raise ValueError("Call fit() first.")
        return self.model.get_cell_effects()

    def get_cell_effects_df(self) -> List[Dict]:
        """
        Get cell effects as a list of dictionaries (easy to convert to DataFrame).

        Returns
        -------
        List of dicts with keys:
            cell_idx, system_a, system_b, system_a_name, system_b_name,
            n_edges, intercept, disease_effect
        """
        alpha = self.get_cell_effects()

        if self.cells is None or self.edges_per_cell is None:
            raise RuntimeError("Preprocessing failed: cells/edges_per_cell not set.")

        n_edges = np.array([len(ec) for ec in self.edges_per_cell])

        results: List[Dict] = []
        for c in range(len(self.cells)):
            a, b = self.cells[c]
            a = int(a)
            b = int(b)
            results.append(
                {
                    "cell_idx": int(c),
                    "system_a": a,
                    "system_b": b,
                    "system_a_name": self.SYSTEM_NAMES.get(a, f"System {a}"),
                    "system_b_name": self.SYSTEM_NAMES.get(b, f"System {b}"),
                    "n_edges": int(n_edges[c]),
                    "intercept": float(alpha[c, 0]),
                    "disease_effect": float(alpha[c, 1]),
                }
            )

        return results

    def get_random_effects_covariance(self) -> np.ndarray:
        """Get U (random effects covariance), shape (C, C)."""
        if self.model is None:
            raise ValueError("Call fit() first.")
        return self.model.get_random_effects_covariance()

    def get_residual_variances(self) -> np.ndarray:
        """Get diagonal of V (residual variances), shape (E,)."""
        if self.model is None:
            raise ValueError("Call fit() first.")
        return self.model.get_residual_variances()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def summary(self) -> str:
        """Return a textual summary of the fitted model."""
        if self.model is None:
            return "Model not fitted. Call fit() first."

        alpha = self.get_cell_effects()
        U = self.get_random_effects_covariance()
        V = self.get_residual_variances()

        diag_U = np.diag(U)

        lines: List[str] = [
            "=" * 60,
            "Graph-Aware Mixed Effects Model Summary",
            "=" * 60,
            f"Subjects: {self.model.N}",
            f"Edges: {self.model.E}",
            f"Cells: {self.model.C}",
            f"Covariates: {self.model.p} (intercept + disease)",
            "",
            f"Converged in {len(self.model.log_likelihood_history)} iterations",
            f"Final log-likelihood: {self.model.log_likelihood_history[-1]:.2f}",
            "",
            "Variance components:",
            f"  U (random effects) diagonal: "
            f"mean={diag_U.mean():.4f}, "
            f"range=[{diag_U.min():.4f}, {diag_U.max():.4f}]",
            f"  V (residual) diagonal: "
            f"mean={V.mean():.4f}, "
            f"range=[{V.min():.4f}, {V.max():.4f}]",
            "",
            "Cell effects (α):",
            f"  Intercept: mean={alpha[:, 0].mean():.4f}, "
            f"std={alpha[:, 0].std():.4f}",
            f"  Disease effect: mean={alpha[:, 1].mean():.4f}, "
            f"std={alpha[:, 1].std():.4f}",
            "",
            "Top 5 cells by |disease effect|:",
        ]

        if self.cells is not None:
            disease_effects = alpha[:, 1]
            top_idx = np.argsort(np.abs(disease_effects))[::-1][:5]

            for rank, c in enumerate(top_idx, 1):
                a, b = self.cells[c]
                a = int(a)
                b = int(b)
                effect = disease_effects[c]
                direction = "↑" if effect > 0 else "↓"
                lines.append(
                    f"  {rank}. ({a},{b}) "
                    f"{self.SYSTEM_NAMES.get(a, f'System {a}')[:20]} - "
                    f"{self.SYSTEM_NAMES.get(b, f'System {b}')[:20]}: "
                    f"{effect:+.4f} {direction}"
                )

        lines.append("=" * 60)

        return "\n".join(lines)
