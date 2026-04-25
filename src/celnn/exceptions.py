"""Public exception module."""

from .core.exceptions import (
    BackendError,
    CelNNError,
    OptionalDependencyError,
    ShapeMismatchError,
    SolverError,
    TemplateValidationError,
)

__all__ = [
    "CelNNError",
    "ShapeMismatchError",
    "TemplateValidationError",
    "SolverError",
    "BackendError",
    "OptionalDependencyError",
]
