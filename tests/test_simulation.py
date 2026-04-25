import numpy as np

from celnn import CellularNetwork, SimulationConfig


def test_simulation_trajectory_shapes():
    signal = np.ones(6)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    result = net.run(
        SimulationConfig(t_end=0.3, dt=0.1, return_trajectory=True)
    )
    assert result.trajectory_state is not None
    assert result.trajectory_output is not None
    assert result.trajectory_state.shape == (4, 6)
    assert result.trajectory_output.shape == (4, 6)
    assert result.time.shape == (4,)


def test_store_every_reduces_trajectory_length():
    signal = np.ones(6)
    net = CellularNetwork(
        input=signal,
        feedback=[0.0, 0.0, 0.0],
        control=[0.0, 1.0, 0.0],
        activation="identity",
    )
    result = net.run(
        SimulationConfig(
            t_end=0.4, dt=0.1, return_trajectory=True, store_every=2
        )
    )
    assert result.trajectory_state is not None
    assert result.trajectory_state.shape[0] == 3
