"""NumPy backend with optional SciPy acceleration."""

from __future__ import annotations

import numpy as np

from ..core.boundary import pad_kwargs, scipy_mode
from ..core.exceptions import BackendError

try:  # pragma: no cover - optional dependency branch
    from scipy.ndimage import convolve as scipy_convolve
except ImportError:  # pragma: no cover - optional dependency branch
    scipy_convolve = None


class NumPyBackend:
    """Backend implementation for NumPy arrays."""

    name = "numpy"

    def aggregate_local(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        *,
        mode: str,
        cval: float = 0.0,
    ) -> np.ndarray:
        """Aggregate local neighborhoods using stencil-aligned weights."""
        array = np.asarray(values, dtype=float)
        kernel = np.asarray(weights, dtype=float)
        if array.ndim != kernel.ndim:
            raise BackendError(
                "Input and template dimensionality must match "
                "for local aggregation. "
                f"Got array ndim={array.ndim}, weights ndim={kernel.ndim}."
            )

        if scipy_convolve is not None:
            flipped = np.flip(kernel)
            return np.asarray(
                scipy_convolve(
                    array, flipped, mode=scipy_mode(mode), cval=float(cval)
                ),
                dtype=float,
            )

        if array.ndim == 1:
            return self._aggregate_1d(array, kernel, mode=mode, cval=cval)
        if array.ndim == 2:
            return self._aggregate_2d(array, kernel, mode=mode, cval=cval)
        raise BackendError(
            "The NumPy fallback currently supports only 1D and 2D arrays. "
            "Install celnn[scipy] for robust ND aggregation."
        )

    def _aggregate_1d(
        self, array: np.ndarray, kernel: np.ndarray, *, mode: str, cval: float
    ) -> np.ndarray:
        radius = kernel.shape[0] // 2
        padded = np.pad(array, (radius, radius), **pad_kwargs(mode, cval))
        result = np.zeros_like(array, dtype=float)
        for index, weight in enumerate(kernel):
            result += float(weight) * padded[index : index + array.shape[0]]
        return result

    def _aggregate_2d(
        self, array: np.ndarray, kernel: np.ndarray, *, mode: str, cval: float
    ) -> np.ndarray:
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = np.pad(
            array, ((pad_y, pad_y), (pad_x, pad_x)), **pad_kwargs(mode, cval)
        )
        result = np.zeros_like(array, dtype=float)
        height, width = array.shape
        for row in range(kernel.shape[0]):
            row_slice = slice(row, row + height)
            for col in range(kernel.shape[1]):
                col_slice = slice(col, col + width)
                result += (
                    float(kernel[row, col]) * padded[row_slice, col_slice]
                )
        return result


NUMPY_BACKEND = NumPyBackend()


def get_default_backend() -> NumPyBackend:
    """Return the default backend instance."""
    return NUMPY_BACKEND
