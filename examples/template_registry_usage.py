"""Register, retrieve, and use a template."""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.templates import Template, TemplateRegistry


def main() -> int:
    registry = TemplateRegistry()
    registry.register(
        Template(
            name="custom_edge_detector",
            feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            control=[
                [-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            bias=-1.0,
            description="Custom edge-like detector",
            tags=["image", "edge", "demo"],
        )
    )

    template = registry.get("custom_edge_detector")
    input_grid = np.zeros((16, 16), dtype=float)
    input_grid[4:12, 4:12] = 1.0

    net = CellularNetwork.from_template(
        template=template,
        input=input_grid,
        activation="piecewise_linear",
        boundary="reflect",
    )
    result = net.run(SimulationConfig(t_end=3.0, dt=0.05))
    print("Registered templates:", registry.names())
    print("Output shape:", result.output.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
