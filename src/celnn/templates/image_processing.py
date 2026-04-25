"""Demonstrative 2D image-processing templates.

Several template values are adapted from the PyCNN project by Ankit Aggarwal
(MIT License) and from common Cellular Neural Network examples. They are
provided here as reusable, demonstrative starting points rather than claims of
optimality.
"""

from __future__ import annotations

from ..core.templates import Template

EDGE_DETECTION = Template(
    name="edge_detection",
    feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    control=[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
    bias=-1.0,
    description="Demonstrative 3x3 edge detector inspired by PyCNN.",
    tags=["image", "edge", "demo", "pycnn-inspired"],
    metadata={"source": "PyCNN (MIT), generalized for celnn"},
)

INVERSION = Template(
    name="inversion",
    feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    control=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
    bias=-2.0,
    description="Demonstrative inversion-like operator inspired by PyCNN.",
    tags=["image", "logic", "demo", "pycnn-inspired"],
    metadata={"source": "PyCNN (MIT), generalized for celnn"},
)

CORNER_DETECTION = Template(
    name="corner_detection",
    feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    control=[[-1.0, -1.0, -1.0], [-1.0, 4.0, -1.0], [-1.0, -1.0, -1.0]],
    bias=-5.0,
    description="Demonstrative corner detector inspired by PyCNN.",
    tags=["image", "corner", "demo", "pycnn-inspired"],
    metadata={"source": "PyCNN (MIT), generalized for celnn"},
)

DIAGONAL_LINE_DETECTION = Template(
    name="diagonal_line_detection",
    feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    control=[[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]],
    bias=-4.0,
    description="Demonstrative diagonal-line detector inspired by PyCNN.",
    tags=["image", "line", "demo", "pycnn-inspired"],
    metadata={"source": "PyCNN (MIT), generalized for celnn"},
)

SMOOTHING_DEMO = Template(
    name="smoothing_demo",
    feedback=[[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]],
    control=[[0.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.0]],
    bias=0.0,
    description="Simple smoothing-like template for experimentation.",
    tags=["image", "smoothing", "demo"],
)

SHARPENING_DEMO = Template(
    name="sharpening_demo",
    feedback=[[-0.05, -0.1, -0.05], [-0.1, 1.5, -0.1], [-0.05, -0.1, -0.05]],
    control=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    bias=0.0,
    description="Simple sharpening-like template for experimentation.",
    tags=["image", "sharpening", "demo"],
)
