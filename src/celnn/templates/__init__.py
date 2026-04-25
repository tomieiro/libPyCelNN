"""Template exports and registry utilities."""

from ..core.templates import Template
from .diffusion import ONE_D_DIFFUSION, TWO_D_LAPLACIAN_DEMO
from .image_processing import (
    CORNER_DETECTION,
    DIAGONAL_LINE_DETECTION,
    EDGE_DETECTION,
    INVERSION,
    SHARPENING_DEMO,
    SMOOTHING_DEMO,
)
from .logic import NOT_DEMO, THRESHOLD_DEMO
from .pattern import DIFFUSION_LIKE, LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO
from .registry import TemplateRegistry, builtin_templates

__all__ = [
    "Template",
    "TemplateRegistry",
    "builtin_templates",
    "EDGE_DETECTION",
    "INVERSION",
    "CORNER_DETECTION",
    "DIAGONAL_LINE_DETECTION",
    "SMOOTHING_DEMO",
    "SHARPENING_DEMO",
    "NOT_DEMO",
    "THRESHOLD_DEMO",
    "DIFFUSION_LIKE",
    "LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO",
    "ONE_D_DIFFUSION",
    "TWO_D_LAPLACIAN_DEMO",
]
