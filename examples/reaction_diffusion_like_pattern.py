"""Run a small reaction-diffusion-like pattern experiment."""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.activations import saturated_linear
from celnn.domains.grid import random_grid
from celnn.templates.pattern import DIFFUSION_LIKE
from celnn.visualization.plots import plot_grid


def main() -> int:
    initial = random_grid((64, 64), low=-1.0, high=1.0, seed=3)
    external_input = np.zeros((64, 64), dtype=float)

    net = CellularNetwork(
        input=external_input,
        initial_state=initial,
        feedback=DIFFUSION_LIKE.feedback,
        control=DIFFUSION_LIKE.control,
        bias=DIFFUSION_LIKE.bias,
        activation=saturated_linear,
        boundary="wrap",
    )

    result = net.run(SimulationConfig(t_start=0.0, t_end=20.0, dt=0.1))
    print("Final pattern mean:", float(np.mean(result.output)))
    print("Final pattern std:", float(np.std(result.output)))

    try:
        plot_grid(
            result.output,
            title="Reaction-diffusion-like pattern",
            path="pattern.png",
        )
        print("Saved optional plot to pattern.png")
    except Exception as exc:
        print(f"Plot skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
