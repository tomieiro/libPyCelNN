"""Backend utilities."""

from .numpy_backend import NUMPY_BACKEND, NumPyBackend, get_default_backend
from .protocol import ArrayBackend

__all__ = [
    "ArrayBackend",
    "NumPyBackend",
    "NUMPY_BACKEND",
    "get_default_backend",
]
