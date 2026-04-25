"""Array helper functions."""

from __future__ import annotations

import numpy as np


def to_float_array(value: object) -> np.ndarray:
    """Convert a value to a floating NumPy array."""
    return np.asarray(value, dtype=float)


def center_index(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the center index for an odd-shaped stencil."""
    return tuple(size // 2 for size in shape)


def is_broadcastable(value: object, shape: tuple[int, ...]) -> bool:
    """Return whether a value can be broadcast to a target shape."""
    try:
        np.broadcast_to(np.asarray(value), shape)
    except ValueError:
        return False
    return True
