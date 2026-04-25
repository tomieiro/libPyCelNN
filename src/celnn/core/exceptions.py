"""Custom exception hierarchy for celnn."""


class CelNNError(Exception):
    """Base exception for the package."""


class ShapeMismatchError(CelNNError):
    """Raised when arrays or shapes are incompatible."""


class TemplateValidationError(CelNNError):
    """Raised when a template is malformed or inconsistent."""


class SolverError(CelNNError):
    """Raised when a numerical solver cannot run."""


class BackendError(CelNNError):
    """Raised when a backend cannot satisfy the requested operation."""


class OptionalDependencyError(CelNNError):
    """Raised when an optional dependency is required but unavailable."""
