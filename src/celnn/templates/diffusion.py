"""Diffusion and Laplacian-like demonstrative templates."""

from __future__ import annotations

from ..core.templates import Template

ONE_D_DIFFUSION = Template(
    name="one_d_diffusion",
    feedback=[0.15, 0.70, 0.15],
    control=[0.0, 0.0, 0.0],
    bias=0.0,
    description="Simple 1D smoothing template for diffusion-like experiments.",
    tags=["signal", "diffusion", "demo"],
)

TWO_D_LAPLACIAN_DEMO = Template(
    name="two_d_laplacian_demo",
    feedback=[[0.0, 0.2, 0.0], [0.2, 0.2, 0.2], [0.0, 0.2, 0.0]],
    control=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    bias=0.0,
    description="Laplacian-like stencil packed as a demonstrative template.",
    tags=["grid", "laplacian", "diffusion", "demo"],
)
