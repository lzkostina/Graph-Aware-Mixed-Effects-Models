"""
Block Coordinate Descent for M-step β Update

The paper (Section 2.5, page 2103) states:
"The most time-consuming part of the algorithm is updating β in the M-step which
involves inverting the large matrix Σ_m X_m^T V^{-1} X_m. In a typical neuroimaging
application, the size of this matrix will be in the tens of thousands; for the COBRE
dataset analyzed in Section 3, it is approximately 28,000 × 28,000. To avoid inverting
this matrix, we instead solve for β using a block coordinate descent algorithm."

The full M-step update for β = (α, η) is:
    β̂ = (Σ_m X_m^T V^{-1} X_m)^{-1} Σ_m X_m^T V^{-1} (y_m - Z⟨γ_m⟩)

Block coordinate descent updates blocks iteratively:
1. Update α (all cells) - this is C*p parameters
2. Update η for each cell c - this is (n_c - 1)*p parameters per cell

Key insight: Due to the structure, η from different cells don't interact,
so we can update η cell-by-cell, never needing to invert a matrix larger
than (n_c * p) × (n_c * p) for the largest cell (~1700 edges → ~3400 × 3400).

Even better: We can use the Woodbury identity and the fact that I_αη = 0
to update α and η separately with proper GLS weighting.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class BlockCDConfig:
    """Configuration for block coordinate descent."""
    max_outer_iter: int = 100      # EM iterations
    max_inner_iter: int = 20       # Inner loop iterations (V, β updates)
    max_cd_iter: int = 50          # Coordinate descent iterations for η
    outer_tol: float = 1e-4        # EM convergence tolerance
    inner_tol: float = 1e-6        # Inner loop tolerance
    cd_tol: float = 1e-8           # Coordinate descent tolerance
    min_variance: float = 1e-8     # Minimum variance
    verbose: bool = False


class GraphAwareEM_BlockCD:
    """
    EM Algorithm with Block Coordinate Descent for the M-step.

    This avoids inverting the full ~28,000 × 28,000 matrix by:
    1. Using the fact that α and η are orthogonal (I_αη = 0)
    2. Updating η cell-by-cell using coordinate descent

    Model:
        y_{m,e} = x_m^T α^{c(e)} + x_m^T η_e + γ_m^{c(e)} + ε_{m,e}

    Using ζ-reparameterization:
        y_{m,e} = ζ_m^{c(e)} + x_m^T η_e + ε_{m,e}
        ζ_m^{(c)} = x_m^T α^{(c)} + γ_m^{(c)}
    """

    def __init__(
        self,
        Y: np.ndarray,              # (N, E) edge weights
        X: np.ndarray,              # (N, p) covariates
        cell_id_of_edge: np.ndarray,  # (E,) edge to cell mapping
        edges_per_cell: List[np.ndarray],  # edge indices per cell
        config: Optional[BlockCDConfig] = None,
    ):
        self.Y = Y
        self.X = X
        self.cell_id_of_edge = cell_id_of_edge
        self.edges_per_cell = edges_per_cell

        self.N, self.E = Y.shape
        self.p = X.shape[1]
        self.C = len(edges_per_cell)

        self.config = config or BlockCDConfig()

        # Precompute X^T X and its inverse (used repeatedly)
        self.XtX = X.T @ X
        self.XtX_inv = np.linalg.inv(self.XtX)

        # State variables
        self.alpha = None    # (C, p)
        self.eta = None      # (E, p)
        self.U = None        # (C, C)
        self.V_diag = None   # (E,)
        self.zeta_mean = None  # (N, C)
        self.zeta_cov = None   # (C, C)

        self.log_likelihood_history = []

    def initialize(self):
        """Initialize parameters."""
        N, E, C, p = self.N, self.E, self.C, self.p

        # Initialize α by OLS on cell means
        cell_means = np.zeros((N, C))
        for c, edge_idx in enumerate(self.edges_per_cell):
            if len(edge_idx) > 0:
                cell_means[:, c] = self.Y[:, edge_idx].mean(axis=1)

        self.alpha = (self.XtX_inv @ self.X.T @ cell_means).T  # (C, p)

        # Initialize η to zero (satisfies constraint)
        self.eta = np.zeros((E, p))

        # Initialize V from residuals
        #fitted = self.X @ self.alpha[self.cell_id_of_edge].T  # (N, E) - wrong shape
        # Actually: for each edge e, fitted[m, e] = x_m^T α^{c(e)}
        #fitted = np.zeros((N, E))
        #for e in range(E):
        #    c = self.cell_id_of_edge[e]
        #    fitted[:, e] = self.X @ self.alpha[c]
        # mu_cells: (N, C)
        mu_cells = self.X @ self.alpha.T
        # expand to edges: (N, E)
        fitted = mu_cells[:, self.cell_id_of_edge]


        residuals = self.Y - fitted
        self.V_diag = np.maximum(np.var(residuals, axis=0), self.config.min_variance)

        # Initialize U from cell mean residuals
        cell_resid = cell_means - self.X @ self.alpha.T
        self.U = np.cov(cell_resid.T) + 0.01 * np.eye(C)

        # Initialize posterior
        self.zeta_mean = cell_means.copy()
        self.zeta_cov = self.U.copy()

        if self.config.verbose:
            print(f"Initialized: α {self.alpha.shape}, η {self.eta.shape}")
            print(f"             U {self.U.shape}, V {self.V_diag.shape}")

    # =========================================================================
    # E-STEP
    # =========================================================================

    def e_step(self):
        """
        E-step: Compute posterior of ζ given current parameters.

        ⟨ζ_m⟩ = Σ_ζ (U^{-1} μ_m + s_m)
        Σ_ζ = (U^{-1} + diag(w))^{-1}

        where:
            μ_m^{(c)} = x_m^T α^{(c)}
            w[c] = Σ_{e in c} 1/v_e
            s_m^{(c)} = Σ_{e in c} (y_{m,e} - x_m^T η_e) / v_e
        """
        N, C = self.N, self.C

        # Precision weights per cell
        w = np.zeros(C)
        for c, edge_idx in enumerate(self.edges_per_cell):
            w[c] = np.sum(1.0 / self.V_diag[edge_idx])

        # U inverse
        try:
            U_inv = np.linalg.inv(self.U)
        except np.linalg.LinAlgError:
            U_inv = np.linalg.pinv(self.U)

        # Posterior covariance (same for all m)
        M = U_inv + np.diag(w)
        self.zeta_cov = np.linalg.inv(M)

        # Prior means
        mu = self.X @ self.alpha.T  # (N, C)

        # Residuals after removing η
        eta_contrib = self.X @ self.eta.T  # (N, E)
        residuals = self.Y - eta_contrib

        # Precision-weighted sums per cell
        s = np.zeros((N, C))
        for c, edge_idx in enumerate(self.edges_per_cell):
            s[:, c] = np.sum(residuals[:, edge_idx] / self.V_diag[edge_idx], axis=1)

        # Posterior means
        rhs = (U_inv @ mu.T).T + s  # (N, C)
        self.zeta_mean = (self.zeta_cov @ rhs.T).T  # (N, C)

    # =========================================================================
    # M-STEP with Block Coordinate Descent
    # =========================================================================

    def m_step(self):
        """
        M-step with block coordinate descent for β = (α, η).

        1. Update U from posterior
        2. Inner loop: alternate V, α, η updates until convergence
        """
        self._update_U()
        self._inner_loop_block_cd()

    def _update_U(self):
        """Update random effects covariance U."""
        mu = self.X @ self.alpha.T  # (N, C)
        diff = self.zeta_mean - mu
        U_new = self.zeta_cov + (diff.T @ diff) / self.N

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(U_new)
        if np.min(eigvals) < self.config.min_variance:
            U_new += (self.config.min_variance - np.min(eigvals) + 1e-6) * np.eye(self.C)

        self.U = U_new

    def _inner_loop_block_cd(self):
        """
        Inner loop using block coordinate descent.

        Updates V, α, η iteratively until convergence.
        """
        for inner_iter in range(self.config.max_inner_iter):
            alpha_old = self.alpha.copy()
            eta_old = self.eta.copy()

            # Update V (diagonal residual variances)
            self._update_V()

            # Update α (cell-level effects) - uses GLS formula
            self._update_alpha_gls()

            # Update η (edge-level effects) - weighted block coordinate descent
            #self._update_eta_weighted_cd()  # Uses V^{-1} weighting
            self._update_eta_block_cd()

            # Check convergence
            alpha_change = np.max(np.abs(self.alpha - alpha_old))
            eta_change = np.max(np.abs(self.eta - eta_old))

            if max(alpha_change, eta_change) < self.config.inner_tol:
                break

    def _update_V(self):
        """Update residual variances V."""
        # Expand zeta to edge level
        zeta_expanded = self.zeta_mean[:, self.cell_id_of_edge]  # (N, E)

        # Edge effect contribution
        eta_contrib = self.X @ self.eta.T  # (N, E)

        # Residuals
        residuals = self.Y - zeta_expanded - eta_contrib

        # Variance per edge
        V_new = np.mean(residuals ** 2, axis=0)

        # Add posterior variance contribution
        for c, edge_idx in enumerate(self.edges_per_cell):
            V_new[edge_idx] += self.zeta_cov[c, c]

        self.V_diag = np.maximum(V_new, self.config.min_variance)

    def _update_alpha_gls(self):
        """
        Update α using GLS formula.

        Since I_αη = 0 (α and η are orthogonal in the GLS sense),
        we can update α independently using the cell-only formula:

        α^{(c)} = (X^T X)^{-1} X^T ⟨ζ^{(c)}⟩

        This is equivalent to the full GLS formula for α.
        """
        self.alpha = (self.XtX_inv @ self.X.T @ self.zeta_mean).T  # (C, p)

    def _update_eta_block_cd(self):
        """
        Update η using block coordinate descent.

        For each cell c, update η_e for all edges e in cell c.
        This respects the sum-to-zero constraint: Σ_{e in c} η_e = 0.

        The update is:
        1. Compute unconstrained optimal η̃_e for each edge
        2. Project to satisfy sum-to-zero constraint

        With V^{-1} weighting, the unconstrained update for edge e is:
            η̃_e = (X^T X)^{-1} X^T (y_{·,e} - ⟨ζ_{·,c(e)}⟩)

        (The 1/v_e weights cancel out for individual edge updates since
        each edge has its own independent parameter.)
        """
        # Expand zeta to edge level
        zeta_expanded = self.zeta_mean[:, self.cell_id_of_edge]  # (N, E)

        # Adjusted response for η regression
        adjusted_Y = self.Y - zeta_expanded  # (N, E)

        # Unconstrained OLS for each edge
        eta_tilde = (self.XtX_inv @ self.X.T @ adjusted_Y).T  # (E, p)

        # Project to satisfy sum-to-zero constraint within each cell
        eta_new = eta_tilde.copy()
        for c, edge_idx in enumerate(self.edges_per_cell):
            if len(edge_idx) > 1:
                cell_mean = eta_tilde[edge_idx].mean(axis=0)
                eta_new[edge_idx] -= cell_mean
            elif len(edge_idx) == 1:
                eta_new[edge_idx] = 0  # Single edge in cell: must be zero

        self.eta = eta_new

    def _update_eta_weighted_cd(self):
        """
        Alternative: Update η with proper V^{-1} weighting using coordinate descent.

        This is more accurate when V varies significantly across edges.

        For cell c, solve:
            min Σ_m Σ_{e in c} (y_{m,e} - ⟨ζ_m^{(c)}⟩ - x_m^T η_e)^2 / v_e
            s.t. Σ_{e in c} η_e = 0

        Using Lagrange multipliers or iterative projection.
        """
        for c, edge_idx in enumerate(self.edges_per_cell):
            n_c = len(edge_idx)
            if n_c <= 1:
                if n_c == 1:
                    self.eta[edge_idx[0]] = 0
                continue

            # Get data for this cell
            Y_c = self.Y[:, edge_idx]  # (N, n_c)
            V_c = self.V_diag[edge_idx]  # (n_c,)
            zeta_c = self.zeta_mean[:, c]  # (N,)

            # Adjusted response
            adjusted_Y_c = Y_c - zeta_c[:, np.newaxis]  # (N, n_c)

            # Weighted unconstrained update
            # For edge e: η̃_e = (X^T W_e X)^{-1} X^T W_e (y_{·,e} - ⟨ζ_{·,c}⟩)
            # where W_e = I_N / v_e (but since v_e is scalar, this simplifies)
            eta_tilde_c = (self.XtX_inv @ self.X.T @ adjusted_Y_c).T  # (n_c, p)

            # Weighted projection to satisfy constraint
            # The constraint-satisfying solution minimizes Σ_e ||η_e - η̃_e||^2 / v_e
            # subject to Σ_e η_e = 0
            #
            # Using Lagrange: η_e = η̃_e - λ, where λ = (Σ_e η̃_e / v_e) / (Σ_e 1/v_e)

            precision = 1.0 / V_c  # (n_c,)
            total_precision = np.sum(precision)

            # Weighted mean of η̃
            weighted_mean = np.sum(eta_tilde_c * precision[:, np.newaxis], axis=0) / total_precision

            # Project
            eta_c = eta_tilde_c - weighted_mean

            self.eta[edge_idx] = eta_c

    # =========================================================================
    # LOG-LIKELIHOOD
    # =========================================================================

    def compute_log_likelihood(self) -> float:
        """Compute marginal log-likelihood."""
        N, E, C = self.N, self.E, self.C

        # Mean: X @ α for each edge
        mean = np.zeros((N, E))
        for e in range(E):
            c = self.cell_id_of_edge[e]
            mean[:, e] = self.X @ (self.alpha[c] + self.eta[e])

        # Covariance: V + Z U Z^T (block structure)
        # Log-likelihood = -0.5 * (log|Σ| + (y-μ)^T Σ^{-1} (y-μ))

        # Using Woodbury for efficient computation
        w = np.zeros(C)
        for c, edge_idx in enumerate(self.edges_per_cell):
            w[c] = np.sum(1.0 / self.V_diag[edge_idx])

        try:
            U_inv = np.linalg.inv(self.U)
        except:
            U_inv = np.linalg.pinv(self.U)

        M = U_inv + np.diag(w)
        M_inv = np.linalg.inv(M)

        # Log determinant: log|Σ| = log|V| + log|U| + log|M|
        log_det_V = np.sum(np.log(self.V_diag))
        log_det_U = np.linalg.slogdet(self.U)[1]
        log_det_M = np.linalg.slogdet(M)[1]
        log_det_Sigma = log_det_V + log_det_U + log_det_M

        # Quadratic form using Woodbury
        residuals = self.Y - mean

        # V^{-1} residuals
        V_inv_resid = residuals / self.V_diag  # (N, E)

        # Sum per cell
        s = np.zeros((N, C))
        for c, edge_idx in enumerate(self.edges_per_cell):
            s[:, c] = np.sum(V_inv_resid[:, edge_idx], axis=1)

        # Quadratic form: r^T V^{-1} r - s^T M^{-1} s
        quad_V = np.sum(residuals * V_inv_resid)
        quad_correction = np.sum(s * (M_inv @ s.T).T)
        quad_form = quad_V - quad_correction

        ll = -0.5 * (N * E * np.log(2 * np.pi) + N * log_det_Sigma + quad_form)

        return ll

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(self):
        """Run the EM algorithm with block coordinate descent."""
        if self.alpha is None:
            self.initialize()

        self.log_likelihood_history = []
        prev_ll = -np.inf

        for iteration in range(self.config.max_outer_iter):
            # E-step
            self.e_step()

            # M-step with block CD
            self.m_step()

            # Log-likelihood
            ll = self.compute_log_likelihood()
            self.log_likelihood_history.append(ll)

            if self.config.verbose:
                print(f"Iter {iteration+1:3d}: LL = {ll:.4f}, change = {ll - prev_ll:.6f}")

            if abs(ll - prev_ll) < self.config.outer_tol:
                if self.config.verbose:
                    print(f"Converged after {iteration+1} iterations")
                break

            if ll < prev_ll - 1e-6:
                warnings.warn(f"Log-likelihood decreased: {prev_ll:.4f} -> {ll:.4f}")

            prev_ll = ll

        return self

    # =========================================================================
    # RESULTS
    # =========================================================================

    def get_cell_effects(self) -> np.ndarray:
        return self.alpha.copy()

    def get_edge_effects(self) -> np.ndarray:
        return self.eta.copy()

    def get_random_effects_covariance(self) -> np.ndarray:
        return self.U.copy()

    def get_residual_variances(self) -> np.ndarray:
        return self.V_diag.copy()

    def get_posterior_means(self) -> np.ndarray:
        return self.zeta_mean.copy()

    def get_posterior_covariance(self) -> np.ndarray:
        return self.zeta_cov.copy()

    def compute_gls_standard_errors(self) -> np.ndarray:
        """
        Compute GLS standard errors for α (cell-level effects).

        Uses the proper formula:
            Var(α̂) = (Σ_m X_m^T Σ^{-1} X_m)^{-1}

        where Σ = V + Z U Z^T.

        Using Woodbury identity for efficiency.

        Returns
        -------
        se : (C, p) array
            Standard errors for α
        """
        # Precision weights per cell
        w = np.zeros(self.C)
        for c, edge_idx in enumerate(self.edges_per_cell):
            w[c] = np.sum(1.0 / self.V_diag[edge_idx])

        # U inverse
        try:
            U_inv = np.linalg.inv(self.U)
        except np.linalg.LinAlgError:
            U_inv = np.linalg.pinv(self.U)

        # M = U^{-1} + diag(w)
        M = U_inv + np.diag(w)
        M_inv = np.linalg.inv(M)

        # W = diag(w) - M_inv ⊙ (w w^T)
        W = np.diag(w) - M_inv * np.outer(w, w)

        # Information matrix for α: (C*p) × (C*p)
        info = np.zeros((self.C * self.p, self.C * self.p))
        for c1 in range(self.C):
            for c2 in range(self.C):
                info[c1*self.p:(c1+1)*self.p, c2*self.p:(c2+1)*self.p] = W[c1, c2] * self.XtX

        # Variance
        var = np.linalg.inv(info)

        # Extract SE
        se = np.zeros((self.C, self.p))
        for c in range(self.C):
            se[c] = np.sqrt(np.diag(var[c*self.p:(c+1)*self.p, c*self.p:(c+1)*self.p]))

        return se