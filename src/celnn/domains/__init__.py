"""Domain-specific helpers kept separate from the numerical core."""

from .grid import checkerboard_grid, coordinate_grid, impulse_grid, random_grid
from .signal import (
    generate_noisy_sine,
    generate_sine_wave,
    normalize_signal,
    plot_signal,
    plot_signal_comparison,
)

__all__ = [
    "checkerboard_grid",
    "coordinate_grid",
    "generate_noisy_sine",
    "generate_sine_wave",
    "impulse_grid",
    "normalize_signal",
    "plot_signal",
    "plot_signal_comparison",
    "random_grid",
]
