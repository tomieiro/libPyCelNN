import numpy as np

from celnn.domains.signal import (
    generate_noisy_sine,
    generate_sine_wave,
    normalize_signal,
)


def test_signal_generation_shapes():
    sine = generate_sine_wave(samples=32, cycles=2.0)
    noisy = generate_noisy_sine(samples=32, cycles=2.0, seed=1)
    assert sine.shape == (32,)
    assert noisy.shape == (32,)


def test_signal_normalization_range():
    signal = np.array([2.0, 4.0, 6.0])
    normalized = normalize_signal(signal)
    assert np.isclose(normalized.min(), -1.0)
    assert np.isclose(normalized.max(), 1.0)
