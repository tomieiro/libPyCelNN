"""Define and use a custom activation function."""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig


def softsign(x: np.ndarray) -> np.ndarray:
    """A smooth bounded activation."""
    array = np.asarray(x, dtype=float)
    return array / (1.0 + np.abs(array))


def main() -> int:
    signal = np.linspace(-1.0, 1.0, 64)

    net = CellularNetwork(
        input=signal,
        feedback=np.array([0.1, 1.0, 0.1]),
        control=np.array([0.0, 1.0, 0.0]),
        bias=0.0,
        activation=softsign,
        boundary="nearest",
    )

    result = net.run(SimulationConfig(t_end=2.0, dt=0.05))
    print(
        "Output min/max:",
        float(result.output.min()),
        float(result.output.max()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
