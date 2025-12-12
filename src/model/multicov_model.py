"""
Extended Graph-Aware Mixed Effects Model with Multiple Covariates

The base model:
    y_{m,e}^{(c)} = x_m^T α^{(c)} + x_m^T η_e^{(c)} + γ_m^{(c)} + ε_{m,e}^{(c)}

supports any number of covariates in x_m. This module provides:
1. Proper covariate handling (centering, standardization, interactions)
2. Hypothesis testing for each covariate
3. Multiple testing correction
4. Model comparison (nested models)

Example covariates:
- Intercept (always included)
- Disease status (binary: 0/1)
- Age (continuous, centered)
- Gender (binary: 0/1)
- Cognitive scores (continuous)
- Disease × Age interaction
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class CovariateInfo:
    """Information about a covariate."""
    name: str
    index: int
    is_binary: bool = False
    is_interaction: bool = False
    components: Optional[Tuple[str, str]] = None  # For interactions


class MultiCovariateMEM:
    """
    Graph-Aware Mixed Effects Model with Multiple Covariates.

    Wraps GraphAwareEM_BlockCD with additional functionality for:
    - Covariate preprocessing
    - Per-covariate hypothesis testing
    - Multiple testing correction
    - Model diagnostics
    """

    def __init__(
            self,
            Y: np.ndarray,
            cell_id_of_edge: np.ndarray,
            edges_per_cell: List[np.ndarray],
            cells: np.ndarray,
    ):
        """
        Initialize with data structure (covariates added separately).

        Parameters
        ----------
        Y : (N, E) array
            Edge weights
        cell_id_of_edge : (E,) array
            Edge to cell mapping
        edges_per_cell : list of arrays
            Edge indices per cell
        cells : (C, 2) array
            Cell definitions
        """
        self.Y = Y
        self.N, self.E = Y.shape
        self.cell_id_of_edge = cell_id_of_edge
        self.edges_per_cell = edges_per_cell
        self.cells = cells
        self.C = len(cells)

        # Covariate storage
        self.covariates: Dict[str, np.ndarray] = {}
        self.covariate_info: Dict[str, CovariateInfo] = {}
        self.X_cov: Optional[np.ndarray] = None
        self.covariate_names: List[str] = []

        # Model results
        self.model = None
        self.is_fitted = False

    def add_intercept(self):
        """Add intercept (column of ones)."""
        if 'intercept' in self.covariates:
            return self  # already have it
        self.covariates['intercept'] = np.ones(self.N)
        self.covariate_info['intercept'] = CovariateInfo(
            name='intercept', index=len(self.covariate_names), is_binary=False
        )
        self.covariate_names.append('intercept')
        return self

    def add_covariate(
            self,
            name: str,
            values: np.ndarray,
            center: bool = False,
            standardize: bool = False,
            is_binary: bool = False,
    ):
        """
        Add a covariate to the model.

        Parameters
        ----------
        name : str
            Name of the covariate
        values : (N,) array
            Covariate values
        center : bool
            Whether to center (subtract mean)
        standardize : bool
            Whether to standardize (center + divide by std)
        is_binary : bool
            Whether this is a binary (0/1) covariate
        """
        values = np.asarray(values).flatten()
        assert len(values) == self.N, f"Expected {self.N} values, got {len(values)}"

        if standardize:
            values = (values - np.mean(values)) / np.std(values)
        elif center:
            values = values - np.mean(values)

        self.covariates[name] = values
        self.covariate_info[name] = CovariateInfo(
            name=name,
            index=len(self.covariate_names),
            is_binary=is_binary,
        )
        self.covariate_names.append(name)
        return self

    def add_interaction(self, name1: str, name2: str, interaction_name: Optional[str] = None):
        """
        Add an interaction term between two existing covariates.

        Parameters
        ----------
        name1, name2 : str
            Names of covariates to interact
        interaction_name : str, optional
            Name for the interaction (default: "name1:name2")
        """
        assert name1 in self.covariates, f"Covariate '{name1}' not found"
        assert name2 in self.covariates, f"Covariate '{name2}' not found"

        if interaction_name is None:
            interaction_name = f"{name1}:{name2}"

        interaction = self.covariates[name1] * self.covariates[name2]

        self.covariates[interaction_name] = interaction
        self.covariate_info[interaction_name] = CovariateInfo(
            name=interaction_name,
            index=len(self.covariate_names),
            is_binary=False,
            is_interaction=True,
            components=(name1, name2),
        )
        self.covariate_names.append(interaction_name)
        return self

    def build_design_matrix(self) -> np.ndarray:
        """Build the design matrix from added covariates."""
        if 'intercept' not in self.covariates:
            warnings.warn("No intercept added. Consider calling add_intercept() first.")

        X = np.column_stack([self.covariates[name] for name in self.covariate_names])
        self.X_cov = X
        return X

    def fit(
            self,
            max_iter: int = 100,
            tol: float = 1e-4,
            verbose: bool = True,
    ):
        """
        Fit the model.

        Parameters
        ----------
        max_iter : int
            Maximum EM iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress
        """
        # Import here to avoid circular imports
        from src.model.block_descent import GraphAwareEM_BlockCD, BlockCDConfig

        if self.X_cov is None:
            self.build_design_matrix()

        if verbose:
            print(f"Fitting model with {len(self.covariate_names)} covariates:")
            for i, name in enumerate(self.covariate_names):
                info = self.covariate_info[name]
                type_str = "binary" if info.is_binary else ("interaction" if info.is_interaction else "continuous")
                print(f"  {i}: {name} ({type_str})")

        config = BlockCDConfig(
            max_outer_iter=max_iter,
            outer_tol=tol,
            verbose=verbose,
        )

        self.model = GraphAwareEM_BlockCD(
            Y=self.Y,
            X=self.X_cov,
            cell_id_of_edge=self.cell_id_of_edge,
            edges_per_cell=self.edges_per_cell,
            config=config,
        )
        self.model.fit()
        self.is_fitted = True

        return self

    def get_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Get estimated coefficients for each covariate.

        Returns
        -------
        coefficients : dict
            {covariate_name: (C,) array of cell-level effects}
        """
        assert self.is_fitted, "Model not fitted yet"

        alpha = self.model.get_cell_effects()  # (C, p)

        return {
            name: alpha[:, self.covariate_info[name].index]
            for name in self.covariate_names
        }

    def get_standard_errors(self) -> Dict[str, np.ndarray]:
        """
        Get standard errors for each covariate.

        Returns
        -------
        se : dict
            {covariate_name: (C,) array of standard errors}
        """
        assert self.is_fitted, "Model not fitted yet"

        se = self.model.compute_gls_standard_errors()  # (C, p)

        return {
            name: se[:, self.covariate_info[name].index]
            for name in self.covariate_names
        }

    def test_covariate(
            self,
            covariate_name: str,
            correction: str = 'none',
            alpha: float = 0.05,
    ) -> Dict:
        """
        Test a covariate for significance in each cell.

        Parameters
        ----------
        covariate_name : str
            Name of covariate to test
        correction : str
            Multiple testing correction: 'none', 'bonferroni', 'bh' (Benjamini-Hochberg)
        alpha : float
            Significance level

        Returns
        -------
        results : dict
            Test results including z-stats, p-values, significant cells
        """
        assert self.is_fitted, "Model not fitted yet"
        assert covariate_name in self.covariate_names, f"Unknown covariate: {covariate_name}"

        idx = self.covariate_info[covariate_name].index

        alpha_hat = self.model.get_cell_effects()[:, idx]  # (C,)
        se = self.model.compute_gls_standard_errors()[:, idx]  # (C,)

        eps = 1e-12
        se_safe = np.maximum(se, eps)
        z_stats = alpha_hat / se_safe
        p_values = 2 * stats.norm.sf(np.abs(z_stats))

        #z_stats = alpha_hat / se
        #p_values = 2 * stats.norm.sf(np.abs(z_stats))

        # Multiple testing correction
        if correction == 'none':
            p_adjusted = p_values
        elif correction == 'bonferroni':
            p_adjusted = np.minimum(p_values * self.C, 1.0)
        elif correction == 'bh':
            p_adjusted = self._benjamini_hochberg(p_values)
        else:
            raise ValueError(f"Unknown correction: {correction}")

        significant = p_adjusted < alpha

        return {
            'covariate': covariate_name,
            'coefficients': alpha_hat,
            'se': se,
            'z_stats': z_stats,
            'p_values': p_values,
            'p_adjusted': p_adjusted,
            'significant': significant,
            'n_significant': int(np.sum(significant)),
            'correction': correction,
            'alpha': alpha,
        }

    def test_all_covariates(
            self,
            skip_intercept: bool = True,
            correction: str = 'bh',
            alpha: float = 0.05,
    ) -> Dict[str, Dict]:
        """
        Test all covariates.

        Parameters
        ----------
        skip_intercept : bool
            Whether to skip testing the intercept
        correction : str
            Multiple testing correction
        alpha : float
            Significance level

        Returns
        -------
        results : dict
            {covariate_name: test_results}
        """
        results = {}
        for name in self.covariate_names:
            if skip_intercept and name == 'intercept':
                continue
            results[name] = self.test_covariate(name, correction=correction, alpha=alpha)
        return results

    def summary(self, correction: str = 'none', alpha: float = 0.05):
        """Print a summary of results for all covariates."""
        assert self.is_fitted, "Model not fitted yet"

        print("=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)

        print(f"\nData: N={self.N} subjects, E={self.E} edges, C={self.C} cells")
        print(f"Covariates: {len(self.covariate_names)}")

        # Variance components
        U = self.model.get_random_effects_covariance()
        V = self.model.get_residual_variances()
        print(f"\nVariance components:")
        print(
            f"  τ² (random): mean={np.mean(np.diag(U)):.4f}, range=[{np.min(np.diag(U)):.4f}, {np.max(np.diag(U)):.4f}]")
        print(f"  σ² (residual): mean={np.mean(V):.4f}")

        # Test each covariate
        print(f"\nCovariate tests (correction={correction}, α={alpha}):")
        print("-" * 70)
        print(f"{'Covariate':<20} {'Mean coef':>10} {'Mean SE':>10} {'Mean |z|':>10} {'# Sig':>8}")
        print("-" * 70)

        for name in self.covariate_names:
            if name == 'intercept':
                continue

            results = self.test_covariate(name, correction=correction, alpha=alpha)

            print(f"{name:<20} {np.mean(results['coefficients']):>+10.4f} "
                  f"{np.mean(results['se']):>10.4f} "
                  f"{np.mean(np.abs(results['z_stats'])):>10.3f} "
                  f"{results['n_significant']:>8d}")

        print("-" * 70)

    def get_significant_cells(
            self,
            covariate_name: str,
            correction: str = 'none',
            alpha: float = 0.05,
    ) -> List[Tuple[int, int, float, float]]:
        """
        Get list of significant cells for a covariate.

        Returns
        -------
        cells : list of (system_a, system_b, coefficient, p_value)
        """
        results = self.test_covariate(covariate_name, correction=correction, alpha=alpha)

        sig_cells = []
        for c in np.where(results['significant'])[0]:
            a, b = self.cells[c]
            sig_cells.append((
                int(a), int(b),
                float(results['coefficients'][c]),
                float(results['p_adjusted'][c]),
            ))

        # Sort by p-value
        sig_cells.sort(key=lambda x: x[3])
        return sig_cells

    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH adjusted p-values
        adjusted = np.zeros(n)
        adjusted[sorted_idx[-1]] = sorted_p[-1]

        for i in range(n - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(
                adjusted[sorted_idx[i + 1]],
                sorted_p[i] * n / (i + 1)
            )

        return np.minimum(adjusted, 1.0)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fit_multicov_model(
        Y: np.ndarray,
        covariates: Dict[str, np.ndarray],
        cell_id_of_edge: np.ndarray,
        edges_per_cell: List[np.ndarray],
        cells: np.ndarray,
        center_continuous: bool = True,
        max_iter: int = 100,
        verbose: bool = True,
) -> MultiCovariateMEM:
    """
    Convenience function to fit a model with multiple covariates.

    Parameters
    ----------
    Y : (N, E) array
        Edge weights
    covariates : dict
        {name: (N,) array} of covariates
        Binary covariates should have values in {0, 1}
    cell_id_of_edge : (E,) array
        Edge to cell mapping
    edges_per_cell : list
        Edge indices per cell
    cells : (C, 2) array
        Cell definitions
    center_continuous : bool
        Whether to center continuous covariates
    max_iter : int
        Maximum iterations
    verbose : bool
        Print progress

    Returns
    -------
    model : MultiCovariateMEM
        Fitted model
    """
    model = MultiCovariateMEM(
        Y=Y,
        cell_id_of_edge=cell_id_of_edge,
        edges_per_cell=edges_per_cell,
        cells=cells,
    )

    # Add intercept
    model.add_intercept()

    # Add covariates
    for name, values in covariates.items():
        vals = np.asarray(values).flatten()
        uniq = np.unique(vals[~np.isnan(vals)])  # if NaNs ever appear
        is_binary = np.all(np.isin(uniq, [0, 1]))
        #is_binary = set(np.unique(values)).issubset({0, 1, 0.0, 1.0})
        center = center_continuous and not is_binary

        model.add_covariate(
            name=name,
            values=values,
            center=center,
            is_binary=is_binary,
        )

    # Fit
    model.fit(max_iter=max_iter, verbose=verbose)

    return model


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demo with multiple covariates."""
    print("=" * 70)
    print("Multi-Covariate Model Demo")
    print("=" * 70)

    # Import structure builder
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from src.design.cells import map_edges_to_cells
    from src.io.power_groups import make_masks_for_power235

    print("\nBuilding structure...")
    _, _, _, _, sys_labels_235 = make_masks_for_power235()
    cells, _, cell_id_of_edge, edges_by_cell = map_edges_to_cells(sys_labels_235, base=1)

    E = len(cell_id_of_edge)
    C = len(cells)
    N = 124

    # Build edges_per_cell as a list of edge index arrays (0..E-1)
    edges_per_cell = [np.where(cell_id_of_edge == c)[0] for c in range(C)]

    print(f"Structure: N={N}, E={E}, C={C}")

    # Generate synthetic data with multiple effects
    np.random.seed(42)

    # Covariates
    disease = np.array([0] * 70 + [1] * 54)  # Binary
    age = np.random.normal(35, 10, N)  # Continuous
    gender = np.random.binomial(1, 0.5, N)  # Binary
    cognitive_score = np.random.normal(100, 15, N)  # Continuous

    # True effects (different covariates affect different cells)
    alpha_true = np.zeros((C, 5))  # [intercept, disease, age, gender, cognitive]
    alpha_true[:, 0] = np.random.randn(C) * 0.3  # Intercept

    # Disease affects cells 0-12
    alpha_true[:13, 1] = np.random.randn(13) * 0.06

    # Age affects cells 20-30
    alpha_true[20:31, 2] = np.random.randn(11) * 0.003  # Small effect per year

    # Gender affects cells 40-50
    alpha_true[40:51, 3] = np.random.randn(11) * 0.05

    # Cognitive score affects cells 60-70
    alpha_true[60:71, 4] = np.random.randn(11) * 0.002

    # Build covariate matrix
    X_cov = np.column_stack([
        np.ones(N),
        disease,
        (age - age.mean()) / age.std(),
        gender,
        (cognitive_score - cognitive_score.mean()) / cognitive_score.std(),
    ])

    # Variance
    tau2, sigma2 = 0.04, 0.015
    U_true = np.eye(C) * tau2

    # Generate data
    gamma = np.random.multivariate_normal(np.zeros(C), U_true, size=N)
    epsilon = np.random.randn(N, E) * np.sqrt(sigma2)

    Y = np.zeros((N, E))
    for m in range(N):
        for e in range(E):
            c = cell_id_of_edge[e]
            Y[m, e] = X_cov[m] @ alpha_true[c] + gamma[m, c] + epsilon[m, e]

    print(f"\nGenerated data with effects in:")
    print(f"  Disease: cells 0-12 (13 cells)")
    print(f"  Age: cells 20-30 (11 cells)")
    print(f"  Gender: cells 40-50 (11 cells)")
    print(f"  Cognitive: cells 60-70 (11 cells)")

    # Fit model
    print("\n" + "=" * 70)
    print("Fitting model...")
    print("=" * 70)

    model = MultiCovariateMEM(
        Y=Y,
        cell_id_of_edge=cell_id_of_edge,
        edges_per_cell=edges_per_cell,
        cells=cells,
    )

    model.add_intercept()
    model.add_covariate('disease', disease, is_binary=True)
    model.add_covariate('age', age, standardize=True)
    model.add_covariate('gender', gender, is_binary=True)
    model.add_covariate('cognitive', cognitive_score, standardize=True)

    model.fit(max_iter=50, verbose=True)

    # Summary
    print("\n")
    model.summary(correction='none', alpha=0.05)

    # Detailed results for each covariate
    print("\n" + "=" * 70)
    print("Detailed Results")
    print("=" * 70)

    for cov in ['disease', 'age', 'gender', 'cognitive']:
        results = model.test_covariate(cov, correction='none', alpha=0.05)
        print(f"\n{cov}:")
        print(f"  Significant cells: {results['n_significant']}")

        sig_cells = model.get_significant_cells(cov, correction='none', alpha=0.05)
        if sig_cells:
            print(f"  Top 5 by p-value:")
            for a, b, coef, pval in sig_cells[:5]:
                print(f"    ({a},{b}): coef={coef:+.4f}, p={pval:.4f}")

    return model


if __name__ == "__main__":
    model = demo()