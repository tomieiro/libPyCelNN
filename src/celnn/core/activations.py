"""Built-in activation functions for cellular outputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .exceptions import CelNNError


def piecewise_linear(x: np.ndarray) -> np.ndarray:
    """Classic bounded piecewise-linear activation for many CelNN models."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (np.abs(x + 1.0) - np.abs(x - 1.0))


def saturated_linear(x: np.ndarray) -> np.ndarray:
    """Alias-like saturating linear output in ``[-1, 1]``."""
    x = np.asarray(x, dtype=float)
    return np.clip(x, -1.0, 1.0)


def identity(x: np.ndarray) -> np.ndarray:
    """Identity activation."""
    return np.asarray(x, dtype=float)


def tanh_activation(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    x = np.asarray(x, dtype=float)
    return np.tanh(x)


def sigmoid_activation(x: np.ndarray) -> np.ndarray:
    """Logistic activation in ``[0, 1]``."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def sign_activation(x: np.ndarray) -> np.ndarray:
    """Sign activation."""
    x = np.asarray(x, dtype=float)
    return np.sign(x)


def relu_activation(x: np.ndarray) -> np.ndarray:
    """Rectified linear activation."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "piecewise_linear": piecewise_linear,
    "saturated_linear": saturated_linear,
    "identity": identity,
    "tanh_activation": tanh_activation,
    "sigmoid_activation": sigmoid_activation,
    "sign_activation": sign_activation,
    "relu_activation": relu_activation,
}


def resolve_activation(
    name_or_callable: str | Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve an activation specified by name or callable."""
    if callable(name_or_callable):
        return name_or_callable
    key = str(name_or_callable).strip()
    try:
        return ACTIVATIONS[key]
    except KeyError as exc:
        known = ", ".join(sorted(ACTIVATIONS))
        raise CelNNError(
            f"Unknown activation '{name_or_callable}'. "
            f"Known activations: {known}."
        ) from exc


def activation_name(
    name_or_callable: str | Callable[[np.ndarray], np.ndarray],
) -> str | None:
    """Return a stable activation name if available."""
    if isinstance(name_or_callable, str):
        return name_or_callable
    for name, func in ACTIVATIONS.items():
        if func is name_or_callable:
            return name
    return None
