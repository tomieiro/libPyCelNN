"""Run a custom template on a small 2D grid."""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.templates import Template


def main() -> int:
    template = Template(
        name="custom_center_weighted",
        feedback=[[0.05, 0.2, 0.05], [0.2, 1.0, 0.2], [0.05, 0.2, 0.05]],
        control=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        bias=-0.1,
        description="A simple custom experiment template.",
        tags=["demo", "custom"],
    )

    input_grid = np.zeros((32, 32), dtype=float)
    input_grid[12:20, 12:20] = 1.0

    net = CellularNetwork.from_template(
        template=template,
        input=input_grid,
        activation="piecewise_linear",
        boundary="reflect",
        metadata={"example": "custom_template_simulation"},
    )
    result = net.run(SimulationConfig(t_end=4.0, dt=0.05))

    print("Template name:", template.name)
    print("Result mean:", float(np.mean(result.output)))
    print("Metadata:", result.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
