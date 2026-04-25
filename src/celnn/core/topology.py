"""Topology helpers for regular-grid cellular networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import normalize_boundary_mode
from .exceptions import BackendError
from .validation import validate_template_shapes


@dataclass(slots=True)
class RegularGridTopology:
    """Topology descriptor for regular Cartesian grids."""

    shape: tuple[int, ...]
    boundary: str = "constant"
    boundary_value: float = 0.0

    def __post_init__(self) -> None:
        self.shape = tuple(int(axis) for axis in self.shape)
        self.boundary = normalize_boundary_mode(self.boundary)
        if any(axis <= 0 for axis in self.shape):
            raise ValueError(
                f"Grid shape must contain positive extents, got {self.shape}."
            )

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)

    def identity_template(self) -> np.ndarray:
        """Return a centered identity-like template."""
        stencil_shape = (3,) * self.ndim
        template = np.zeros(stencil_shape, dtype=float)
        center = tuple(size // 2 for size in stencil_shape)
        template[center] = 1.0
        return template

    def zero_template(self) -> np.ndarray:
        """Return a zero template with the default stencil shape."""
        return np.zeros((3,) * self.ndim, dtype=float)

    def validate_template(
        self, feedback: np.ndarray, control: np.ndarray
    ) -> None:
        """Validate template compatibility against this topology."""
        validate_template_shapes(feedback, control, self.ndim)

    def numpy_fallback_supported(self) -> bool:
        """Return whether the NumPy fallback supports this dimensionality."""
        return self.ndim in (1, 2)

    def require_numpy_fallback_support(self) -> None:
        """Raise if the current dimensionality exceeds the NumPy fallback."""
        if not self.numpy_fallback_supported():
            raise BackendError(
                "The pure NumPy fallback currently supports only "
                "1D and 2D stencils. "
                "Install celnn[scipy] for robust ND aggregation."
            )
