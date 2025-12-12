"""
EM Algorithm for Graph-Aware Mixed Effects Model

Implementation of Kim, Kessler, and Levina (2023) Section 2.5.

Model (full):
    y_{m,i}^{(c)} = x_m^T α^{(c)} + x_m^T η_i^{(c)} + γ_m^{(c)} + ε_{m,i}^{(c)}

With ζ-reparameterization:
    y_{m,i}^{(c)} = ζ_m^{(c)} + x_m^T η_i^{(c)} + ε_{m,i}^{(c)}

where:
    ζ_m^{(c)} = x_m^T α^{(c)} + γ_m^{(c)}
    ζ_m ~ N(μ_m, U),  μ_m^{(c)} = x_m^T α^{(c)}
    ε_{m,i}^{(c)} ~ N(0, v_e)  (diagonal V)

Parameters:
    α: (C, p) cell-level fixed effects
    η: (E, p) edge-level fixed effects with sum-to-zero constraint per cell
    U: (C, C) cell-level random effect covariance
    V: (E,) diagonal residual variances
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings


@dataclass
class EMConfig:
    """Configuration for EM algorithm."""
    max_iter: int = 100
    tol: float = 1e-4
    max_inner_iter: int = 10
    inner_tol: float = 1e-6
    min_variance: float = 1e-8
    verbose: bool = False


@dataclass
class EMState:
    """
    Current state of EM algorithm parameters.

    Attributes
    ----------
    alpha : (C, p) cell-level fixed effects
    eta : (E, p) edge-level fixed effects (sum to zero within each cell)
    U : (C, C) random effects covariance matrix
    V_diag : (E,) diagonal of residual covariance
    zeta_mean : (N, C) posterior means of ζ_m
    zeta_cov : (C, C) posterior covariance (same for all m)
    """
    alpha: np.ndarray
    eta: np.ndarray
    U: np.ndarray
    V_diag: np.ndarray
    zeta_mean: Optional[np.ndarray] = None
    zeta_cov: Optional[np.ndarray] = None


class GraphAwareEM:
    """
    EM Algorithm for the graph-aware mixed effects model.

    Parameters
    ----------
    Y : (N, E) array
        Edge weight matrix. Y[m, e] is the weight of edge e for subject m.
    X : (N, p) array
        Covariate matrix. X[m, :] are covariates for subject m.
        Typically X = [1, disease_indicator] for the two-sample problem.
    cell_id_of_edge : (E,) array
        Maps each edge to its cell index in [0, C-1].
    edges_per_cell : list of C arrays
        edges_per_cell[c] contains the indices of edges in cell c.
    config : EMConfig, optional
        Algorithm configuration.
    """

    def __init__(
            self,
            Y: np.ndarray,
            X: np.ndarray,
            cell_id_of_edge: np.ndarray,
            edges_per_cell: List[np.ndarray],
            config: Optional[EMConfig] = None,
    ):
        # Store data
        self.Y = np.asarray(Y, dtype=np.float64)
        self.X = np.asarray(X, dtype=np.float64)
        self.cell_id_of_edge = np.asarray(cell_id_of_edge, dtype=np.int32)
        self.edges_per_cell = edges_per_cell
        self.config = config or EMConfig()

        # Extract dimensions
        self.N, self.E = self.Y.shape
        self.N_check, self.p = self.X.shape
        self.C = len(edges_per_cell)

        # Validate inputs
        self._validate_inputs()

        # Precompute useful quantities
        self.n_edges_per_cell = np.array([len(ec) for ec in edges_per_cell])
        self.XtX = self.X.T @ self.X  # (p, p) - used repeatedly
        self.XtX_inv = np.linalg.inv(self.XtX)  # (p, p)

        # State will be initialized by initialize() or fit()
        self.state: Optional[EMState] = None
        self.log_likelihood_history: List[float] = []

    def _validate_inputs(self):
        """Validate input dimensions and values."""
        if self.N != self.N_check:
            raise ValueError(f"Y has {self.N} subjects but X has {self.N_check}")

        if self.cell_id_of_edge.shape != (self.E,):
            raise ValueError(f"cell_id_of_edge shape {self.cell_id_of_edge.shape} != ({self.E},)")

        if self.cell_id_of_edge.min() < 0 or self.cell_id_of_edge.max() >= self.C:
            raise ValueError(f"cell_id_of_edge values must be in [0, {self.C - 1}]")

        if not isinstance(self.edges_per_cell, (list, tuple)):
            raise TypeError(
                f"edges_per_cell must be a list or tuple of arrays, "
                f"got {type(self.edges_per_cell).__name__!r} instead. "
                f"This often means GraphAwareEM.__init__ was called with "
                f"positional arguments in the wrong order."
            )

        # Check edges_per_cell covers all edges exactly once
        all_edges = np.concatenate(self.edges_per_cell)
        if len(all_edges) != self.E or set(all_edges) != set(range(self.E)):
            raise ValueError("edges_per_cell must partition [0, E-1]")

        if not np.isfinite(self.Y).all():
            raise ValueError("Y contains non-finite values")

        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def initialize(self) -> EMState:
        """
        Initialize parameters using OLS and method of moments.

        Following Section 2.5 of the paper:
        1. Initialize α by OLS on cell means
        2. Initialize η = 0
        3. Initialize U from cross-cell covariances of residuals
        4. Initialize V from residual variances

        Returns
        -------
        state : EMState
            Initial parameter state.
        """
        N, E, C, p = self.N, self.E, self.C, self.p

        # -----------------------------------------------------------------
        # Step 1: Compute cell means for each subject
        # y_bar_m^{(c)} = (1/n_c) * sum_{e in cell c} y_{m,e}
        # -----------------------------------------------------------------
        cell_means = np.zeros((N, C))
        for c, edge_idx in enumerate(self.edges_per_cell):
            if len(edge_idx) > 0:
                cell_means[:, c] = self.Y[:, edge_idx].mean(axis=1)

        # -----------------------------------------------------------------
        # Step 2: Initialize α by OLS on cell means
        # α^{(c)} = (X^T X)^{-1} X^T y_bar^{(c)}
        # -----------------------------------------------------------------
        alpha = (self.XtX_inv @ self.X.T @ cell_means).T  # (C, p)

        # -----------------------------------------------------------------
        # Step 3: Initialize η = 0 (edge effects)
        # -----------------------------------------------------------------
        eta = np.zeros((E, p))

        # -----------------------------------------------------------------
        # Step 4: Compute residuals for U initialization
        # r_m^{(c)} = y_bar_m^{(c)} - x_m^T α^{(c)}
        # -----------------------------------------------------------------
        mu = self.X @ alpha.T  # (N, C) predicted cell means
        cell_residuals = cell_means - mu  # (N, C)

        # -----------------------------------------------------------------
        # Step 5: Initialize U as empirical covariance of cell residuals
        # U = (1/(N-1)) * sum_m (r_m - r_bar)(r_m - r_bar)^T
        # -----------------------------------------------------------------
        U = np.cov(cell_residuals.T)  # (C, C)
        if U.ndim == 0:
            U = np.array([[U]])

        # Ensure U is positive definite
        U = self._ensure_positive_definite(U)

        # -----------------------------------------------------------------
        # Step 6: Initialize V (diagonal) from edge-level residual variances
        # -----------------------------------------------------------------
        # Expand mu to edge level using Z
        mu_expanded = mu[:, self.cell_id_of_edge]  # (N, E)
        edge_residuals = self.Y - mu_expanded  # (N, E)

        # Variance per edge, minus the random effect contribution
        V_diag = np.var(edge_residuals, axis=0, ddof=1)  # (E,)

        # Subtract U contribution (approximate)
        for c, edge_idx in enumerate(self.edges_per_cell):
            V_diag[edge_idx] = np.maximum(
                V_diag[edge_idx] - U[c, c],
                self.config.min_variance
            )

        # -----------------------------------------------------------------
        # Create and store state
        # -----------------------------------------------------------------
        self.state = EMState(
            alpha=alpha,
            eta=eta,
            U=U,
            V_diag=V_diag,
        )

        if self.config.verbose:
            print(f"Initialized: α shape {alpha.shape}, η shape {eta.shape}")
            print(f"             U shape {U.shape}, V_diag shape {V_diag.shape}")
            print(f"             U diagonal range: [{np.diag(U).min():.4f}, {np.diag(U).max():.4f}]")
            print(f"             V range: [{V_diag.min():.4f}, {V_diag.max():.4f}]")

        return self.state

    def _ensure_positive_definite(self, M: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive definite by clipping eigenvalues."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, min_eig)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    # =========================================================================
    # E-STEP
    # =========================================================================

    def e_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: Compute posterior distribution of ζ_m given current parameters.

        Using the ζ-parameterization:
            ζ_m ~ N(μ_m, U)  prior
            y_{m,e} | ζ_m ~ N(ζ_m^{c(e)} + x_m^T η_e, v_e)  likelihood

        Posterior:
            ζ_m | y_m ~ N(⟨ζ_m⟩, Σ_ζ)

        With Woodbury identity for efficient computation:
            M = U^{-1} + diag(w)  where w_c = sum_{e in c} 1/v_e
            Σ_ζ = M^{-1}
            ⟨ζ_m⟩ = Σ_ζ (U^{-1} μ_m + s_m)

        where s_m^{(c)} = sum_{e in c} (y_{m,e} - x_m^T η_e) / v_e

        Returns
        -------
        zeta_mean : (N, C) posterior means
        zeta_cov : (C, C) posterior covariance (same for all m)
        """
        N, E, C, p = self.N, self.E, self.C, self.p
        alpha, eta, U, V_diag = self.state.alpha, self.state.eta, self.state.U, self.state.V_diag

        # -----------------------------------------------------------------
        # Step 1: Compute precision weights per cell
        # w_c = sum_{e in cell c} 1/v_e
        # -----------------------------------------------------------------
        w = np.zeros(C)
        for c, edge_idx in enumerate(self.edges_per_cell):
            w[c] = np.sum(1.0 / V_diag[edge_idx])

        # -----------------------------------------------------------------
        # Step 2: Compute U^{-1}
        # -----------------------------------------------------------------
        try:
            U_inv = np.linalg.inv(U)
        except np.linalg.LinAlgError:
            U_inv = np.linalg.pinv(U)

        # -----------------------------------------------------------------
        # Step 3: Compute middle matrix M and its inverse
        # M = U^{-1} + diag(w)
        # Σ_ζ = M^{-1}
        # -----------------------------------------------------------------
        M = U_inv + np.diag(w)

        try:
            zeta_cov = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            zeta_cov = np.linalg.pinv(M)

        # -----------------------------------------------------------------
        # Step 4: Compute prior means μ_m^{(c)} = x_m^T α^{(c)}
        # -----------------------------------------------------------------
        mu = self.X @ alpha.T  # (N, C)

        # -----------------------------------------------------------------
        # Step 5: Compute residuals after edge effects
        # r̃_{m,e} = y_{m,e} - x_m^T η_e
        # -----------------------------------------------------------------
        # η contribution: (N, E) where each row m gets X[m] @ eta.T
        eta_contribution = self.X @ eta.T  # (N, E)
        residuals = self.Y - eta_contribution  # (N, E)

        # -----------------------------------------------------------------
        # Step 6: Compute precision-weighted sums per cell
        # s_m^{(c)} = sum_{e in cell c} r̃_{m,e} / v_e
        # -----------------------------------------------------------------
        s = np.zeros((N, C))
        for c, edge_idx in enumerate(self.edges_per_cell):
            # residuals[:, edge_idx] is (N, n_c)
            # V_diag[edge_idx] is (n_c,)
            s[:, c] = np.sum(residuals[:, edge_idx] / V_diag[edge_idx], axis=1)

        # -----------------------------------------------------------------
        # Step 7: Compute posterior means
        # ⟨ζ_m⟩ = Σ_ζ (U^{-1} μ_m + s_m)
        # -----------------------------------------------------------------
        # U^{-1} @ mu.T gives (C, N), then transpose
        rhs = (U_inv @ mu.T).T + s  # (N, C)
        zeta_mean = (zeta_cov @ rhs.T).T  # (N, C)

        # Store in state
        self.state.zeta_mean = zeta_mean
        self.state.zeta_cov = zeta_cov

        return zeta_mean, zeta_cov

    # =========================================================================
    # M-STEP
    # =========================================================================

    def m_step(self) -> None:
        """
        M-step: Update parameters given posterior of ζ.

        Updates:
        1. U: random effects covariance
        2. Inner loop for (V, α, η) until convergence
        """
        self._update_U()
        self._inner_loop()

    def _update_U(self) -> None:
        """
        Update U (random effects covariance).

        U = Σ_ζ + (1/N) sum_m (⟨ζ_m⟩ - μ_m)(⟨ζ_m⟩ - μ_m)^T

        where μ_m^{(c)} = x_m^T α^{(c)}
        """
        N, C = self.N, self.C
        zeta_mean = self.state.zeta_mean
        zeta_cov = self.state.zeta_cov
        alpha = self.state.alpha

        # Prior means: μ_m^{(c)} = x_m^T α^{(c)}
        mu = self.X @ alpha.T  # (N, C)

        # Deviations from prior mean
        diff = zeta_mean - mu  # (N, C)

        # Empirical covariance of deviations + posterior covariance
        U_new = zeta_cov + (diff.T @ diff) / N

        # Ensure positive definite
        self.state.U = self._ensure_positive_definite(U_new)

    def _inner_loop(self) -> None:
        """
        Inner loop: alternately update V, α, η until convergence.

        This follows Section 2.5 of the paper.
        """
        for inner_iter in range(self.config.max_inner_iter):
            alpha_old = self.state.alpha.copy()

            # Update V (diagonal residual variances)
            self._update_V()

            # Update α (cell-level fixed effects)
            self._update_alpha()

            # Update η (edge-level fixed effects with constraint)
            self._update_eta()

            # Check convergence
            alpha_change = np.max(np.abs(self.state.alpha - alpha_old))
            if alpha_change < self.config.inner_tol:
                break

    def _update_V(self) -> None:
        """
        Update V (diagonal residual variances).

        v_e = (1/N) sum_m (y_{m,e} - ⟨ζ_m^{c(e)}⟩ - x_m^T η_e)^2 + (Σ_ζ)_{c(e),c(e)}

        The second term accounts for uncertainty in ζ.
        """
        N, E = self.N, self.E
        zeta_mean = self.state.zeta_mean
        zeta_cov = self.state.zeta_cov
        eta = self.state.eta

        # Expand zeta_mean to edge level: (N, E)
        zeta_expanded = zeta_mean[:, self.cell_id_of_edge]

        # Edge effect contribution: (N, E)
        eta_contribution = self.X @ eta.T

        # Residuals
        residuals = self.Y - zeta_expanded - eta_contribution  # (N, E)

        # Empirical variance per edge
        V_new = np.mean(residuals ** 2, axis=0)  # (E,)

        # Add posterior variance contribution
        for c, edge_idx in enumerate(self.edges_per_cell):
            V_new[edge_idx] += zeta_cov[c, c]

        # Ensure minimum variance
        self.state.V_diag = np.maximum(V_new, self.config.min_variance)

    def _update_alpha(self) -> None:
        """
        Update α (cell-level fixed effects).

        α^{(c)} = (X^T X)^{-1} X^T ⟨ζ^{(c)}⟩

        This is OLS regression of posterior means on covariates.
        """
        zeta_mean = self.state.zeta_mean  # (N, C)

        # α = (X^T X)^{-1} X^T zeta_mean, then transpose to get (C, p)
        self.state.alpha = (self.XtX_inv @ self.X.T @ zeta_mean).T

    def _update_eta(self) -> None:
        """
        Update η (edge-level fixed effects) with sum-to-zero constraint.

        For each edge e in cell c:
            η̃_e = (X^T X)^{-1} X^T (y_{·,e} - ⟨ζ_{·,c}⟩)

        Then project to satisfy constraint:
            η̂_e = η̃_e - (1/n_c) sum_{e' in c} η̃_{e'}
        """
        N, E, C, p = self.N, self.E, self.C, self.p
        zeta_mean = self.state.zeta_mean

        # Expand zeta to edge level
        zeta_expanded = zeta_mean[:, self.cell_id_of_edge]  # (N, E)

        # Adjusted response for η regression
        adjusted_Y = self.Y - zeta_expanded  # (N, E)

        # OLS for each edge (vectorized)
        # η̃ = (X^T X)^{-1} X^T adjusted_Y
        eta_tilde = (self.XtX_inv @ self.X.T @ adjusted_Y).T  # (E, p)

        # Project to satisfy sum-to-zero constraint within each cell
        eta_new = eta_tilde.copy()
        for c, edge_idx in enumerate(self.edges_per_cell):
            if len(edge_idx) > 0:
                cell_mean = eta_tilde[edge_idx].mean(axis=0)  # (p,)
                eta_new[edge_idx] -= cell_mean

        self.state.eta = eta_new

    # =========================================================================
    # LOG-LIKELIHOOD
    # =========================================================================

    def compute_log_likelihood(self) -> float:
        """
        Compute marginal log-likelihood.

        y_m ~ N(X_m β, Σ) where Σ = V + Z U Z^T

        Using Woodbury identity for efficient computation:
            log|Σ| = log|V| + log|M| + log|U|
            y^T Σ^{-1} y = y^T V^{-1} y - b^T M^{-1} b

        where b_c = sum_{e in c} (y_e - fixed_effects_e) / v_e
        """
        N, E, C = self.N, self.E, self.C
        alpha, eta, U, V_diag = self.state.alpha, self.state.eta, self.state.U, self.state.V_diag

        # -----------------------------------------------------------------
        # Precompute: precision weights and middle matrix
        # -----------------------------------------------------------------
        w = np.zeros(C)
        for c, edge_idx in enumerate(self.edges_per_cell):
            w[c] = np.sum(1.0 / V_diag[edge_idx])

        try:
            U_inv = np.linalg.inv(U)
        except np.linalg.LinAlgError:
            U_inv = np.linalg.pinv(U)

        M = U_inv + np.diag(w)

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # -----------------------------------------------------------------
        # Log determinant of Σ using Woodbury
        # log|Σ| = log|V| + log|M| + log|U|
        # -----------------------------------------------------------------
        log_det_V = np.sum(np.log(V_diag))
        _, log_det_M = np.linalg.slogdet(M)
        _, log_det_U = np.linalg.slogdet(U)
        log_det_Sigma = log_det_V + log_det_M + log_det_U

        # -----------------------------------------------------------------
        # Compute log-likelihood for each subject
        # -----------------------------------------------------------------
        ll = 0.0

        for m in range(N):
            # Mean vector: μ_m^{(c)} = x_m^T α^{(c)}, expanded to edges
            mu_cells = self.X[m] @ alpha.T  # (C,)
            mu_edges = mu_cells[self.cell_id_of_edge]  # (E,)

            # Add edge effects
            eta_contribution = self.X[m] @ eta.T  # (E,)
            mean_m = mu_edges + eta_contribution

            # Centered data
            d = self.Y[m] - mean_m  # (E,)

            # Quadratic form part 1: d^T V^{-1} d
            quad1 = np.sum(d ** 2 / V_diag)

            # Quadratic form part 2: b^T M^{-1} b
            # where b_c = sum_{e in c} d_e / v_e
            b = np.zeros(C)
            for c, edge_idx in enumerate(self.edges_per_cell):
                b[c] = np.sum(d[edge_idx] / V_diag[edge_idx])

            quad2 = b @ M_inv @ b

            # Combined quadratic form
            quad = quad1 - quad2

            # Log-likelihood contribution
            ll += -0.5 * (E * np.log(2 * np.pi) + log_det_Sigma + quad)

        return ll

    # =========================================================================
    # MAIN FIT METHOD
    # =========================================================================

    def fit(self) -> 'GraphAwareEM':
        """
        Run the EM algorithm.

        Returns
        -------
        self : GraphAwareEM
            Fitted model.
        """
        # Initialize if not already done
        if self.state is None:
            self.initialize()

        self.log_likelihood_history = []
        prev_ll = -np.inf

        for iteration in range(self.config.max_iter):
            # E-step
            self.e_step()

            # M-step
            self.m_step()

            # Compute log-likelihood
            ll = self.compute_log_likelihood()
            self.log_likelihood_history.append(ll)

            if self.config.verbose:
                print(f"Iter {iteration + 1:3d}: LL = {ll:.4f}, change = {ll - prev_ll:.6f}")

            # Check convergence
            if abs(ll - prev_ll) < self.config.tol:
                if self.config.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            # Check for decrease (shouldn't happen in EM)
            if ll < prev_ll - 1e-6:
                warnings.warn(f"Log-likelihood decreased at iteration {iteration + 1}: "
                              f"{prev_ll:.6f} -> {ll:.6f}")

            prev_ll = ll

        return self

    # =========================================================================
    # RESULTS ACCESS
    # =========================================================================

    def get_cell_effects(self) -> np.ndarray:
        """Return α (cell-level fixed effects), shape (C, p)."""
        return self.state.alpha.copy()

    def get_edge_effects(self) -> np.ndarray:
        """Return η (edge-level fixed effects), shape (E, p)."""
        return self.state.eta.copy()

    def get_random_effects_covariance(self) -> np.ndarray:
        """Return U (random effects covariance), shape (C, C)."""
        return self.state.U.copy()

    def get_residual_variances(self) -> np.ndarray:
        """Return diagonal of V (residual variances), shape (E,)."""
        return self.state.V_diag.copy()

    def get_posterior_means(self) -> np.ndarray:
        """Return ⟨ζ_m⟩ (posterior means), shape (N, C)."""
        return self.state.zeta_mean.copy()

    def get_posterior_covariance(self) -> np.ndarray:
        """Return Σ_ζ (posterior covariance), shape (C, C)."""
        return self.state.zeta_cov.copy()


# =============================================================================
# CELL-ONLY MODEL (simplified, no edge effects)
# =============================================================================

class GraphAwareEM_CellOnly(GraphAwareEM):
    """
    Simplified EM for cell-only model (no edge effects η).

    Model:
        y_{m,i}^{(c)} = ζ_m^{(c)} + ε_{m,i}^{(c)}

    This is faster and gives identical cell-level inference for α.
    """

    def _update_eta(self) -> None:
        """Skip η update - keep at zero."""
        pass

    def _inner_loop(self) -> None:
        """Simplified inner loop without η updates."""
        for inner_iter in range(self.config.max_inner_iter):
            alpha_old = self.state.alpha.copy()

            # Update V
            self._update_V()

            # Update α
            self._update_alpha()

            # Check convergence
            alpha_change = np.max(np.abs(self.state.alpha - alpha_old))
            if alpha_change < self.config.inner_tol:
                break