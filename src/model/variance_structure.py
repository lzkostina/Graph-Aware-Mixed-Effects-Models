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


