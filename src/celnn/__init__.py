"""Top-level package for celnn."""

from .activations import (
    identity,
    piecewise_linear,
    relu_activation,
    saturated_linear,
    sigmoid_activation,
    sign_activation,
    tanh_activation,
)
from .core.network import CellularNetwork
from .core.result import SimulationResult
from .core.simulation import SimulationConfig
from .core.templates import Template
from .templates.registry import TemplateRegistry

__all__ = [
    "CellularNetwork",
    "SimulationConfig",
    "SimulationResult",
    "Template",
    "TemplateRegistry",
    "identity",
    "piecewise_linear",
    "relu_activation",
    "saturated_linear",
    "sign_activation",
    "sigmoid_activation",
    "tanh_activation",
]

__version__ = "0.1.0"
