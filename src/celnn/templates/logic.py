"""Small logical or threshold-like templates."""

from __future__ import annotations

from ..core.templates import Template

NOT_DEMO = Template(
    name="not_demo",
    feedback=[0.0, 1.0, 0.0],
    control=[0.0, -1.0, 0.0],
    bias=0.0,
    description=(
        "Simple 1D inversion-like template for scalar or signal experiments."
    ),
    tags=["logic", "signal", "demo"],
)

THRESHOLD_DEMO = Template(
    name="threshold_demo",
    feedback=[0.2, 1.0, 0.2],
    control=[0.0, 1.0, 0.0],
    bias=-0.25,
    description="Thresholding-like 1D template for signal experiments.",
    tags=["logic", "signal", "threshold", "demo"],
)
