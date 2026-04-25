"""Pattern-formation oriented demonstrative templates."""

from __future__ import annotations

from ..core.templates import Template

DIFFUSION_LIKE = Template(
    name="diffusion_like",
    feedback=[[0.0, 0.2, 0.0], [0.2, 0.2, 0.2], [0.0, 0.2, 0.0]],
    control=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    bias=0.0,
    description=(
        "Simple local smoothing template reminiscent "
        "of diffusion-like behavior."
    ),
    tags=["pattern", "diffusion", "demo"],
)

LOCAL_EXCITATION_GLOBAL_DAMPING_DEMO = Template(
    name="local_excitation_global_damping_demo",
    feedback=[
        [-0.02, -0.03, -0.04, -0.03, -0.02],
        [-0.03, 0.02, 0.05, 0.02, -0.03],
        [-0.04, 0.05, 1.20, 0.05, -0.04],
        [-0.03, 0.02, 0.05, 0.02, -0.03],
        [-0.02, -0.03, -0.04, -0.03, -0.02],
    ],
    control=[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    bias=-0.1,
    description=(
        "Small excitatory-center template for "
        "pattern-formation demonstrations."
    ),
    tags=["pattern", "reaction-diffusion", "demo"],
)
