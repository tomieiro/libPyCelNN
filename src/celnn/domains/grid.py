"""Helpers for synthetic regular grids."""

from __future__ import annotations

import numpy as np


def random_grid(
    shape: tuple[int, ...],
    *,
    low: float = -1.0,
    high: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a random grid."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape)


def impulse_grid(
    shape: tuple[int, ...],
    *,
    location: tuple[int, ...] | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a grid with a single impulse."""
    grid = np.zeros(shape, dtype=float)
    chosen_location = location or tuple(axis // 2 for axis in shape)
    grid[chosen_location] = amplitude
    return grid


def checkerboard_grid(
    shape: tuple[int, int], *, values: tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """Generate a 2D checkerboard pattern."""
    rows, cols = shape
    yy, xx = np.indices((rows, cols))
    return np.where((yy + xx) % 2 == 0, values[0], values[1]).astype(float)


def coordinate_grid(
    shape: tuple[int, ...], *, normalize: bool = True
) -> np.ndarray:
    """Generate a coordinate grid with coordinates stored on the last axis."""
    coords = np.indices(shape, dtype=float)
    if normalize:
        for axis, size in enumerate(shape):
            if size > 1:
                coords[axis] = (2.0 * coords[axis] / float(size - 1)) - 1.0
            else:
                coords[axis] = 0.0
    return np.moveaxis(coords, 0, -1)
