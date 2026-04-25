import numpy as np

from celnn.activations import (
    identity,
    piecewise_linear,
    relu_activation,
    saturated_linear,
    sigmoid_activation,
    sign_activation,
    tanh_activation,
)


def test_piecewise_linear_bounds_output():
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    y = piecewise_linear(x)
    assert np.all(y <= 1.0)
    assert np.all(y >= -1.0)


def test_saturated_linear_clips():
    x = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(saturated_linear(x), [-1.0, 0.0, 1.0])


def test_other_activations_return_expected_shapes():
    x = np.array([-1.0, 0.0, 1.0])
    assert identity(x).shape == x.shape
    assert tanh_activation(x).shape == x.shape
    assert sigmoid_activation(x).shape == x.shape
    assert sign_activation(x).shape == x.shape
    assert relu_activation(x).shape == x.shape
