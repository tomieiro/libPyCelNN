# Template design guide

## What templates are

A template is a local stencil of coefficients that tells each cell how to combine:

- neighboring outputs through feedback `A`,
- neighboring inputs through control `B`.

In `celnn`, a `Template` bundles `feedback`, `control`, `bias`, optional `initial_state`, plus descriptive metadata.

## How `A` and `B` affect dynamics

- `A` shapes recurrent local interaction. It changes how current outputs influence future evolution.
- `B` shapes how the external input field drives the dynamics.

If `B` is large and `A` is weak, the system behaves more like an input-driven local filter. If `A` is strong, the system becomes more self-organized and dynamical.

## How bias affects thresholds

Bias is the simplest way to shift activation thresholds:

- negative bias suppresses activation,
- positive bias favors activation,
- spatially varying bias can create nonuniform behavior across the domain.

## How to design simple filters

Useful starter heuristics:

- for smoothing, use a positive center and modest positive neighbors,
- for contrast or edge effects, use mixed signs with near-zero total sum,
- for thresholding, combine moderate self-feedback with a bias shift,
- for diffusion-like behavior, spread moderate positive weights over nearby cells.

## How to port PyCNN-style templates

PyCNN exposes image-specific methods such as `edgeDetection(...)` and `cornerDetection(...)`. In `celnn`, port them as named `Template` objects and run them through `CellularNetwork.from_template(...)`.

Example:

```python
from celnn import CellularNetwork, SimulationConfig
from celnn.templates.image_processing import EDGE_DETECTION

net = CellularNetwork.from_template(
    template=EDGE_DETECTION,
    input=my_array,
    activation="piecewise_linear",
    boundary="reflect",
)
```

## How to validate template shape

`celnn` enforces:

- same shape for `feedback` and `control`,
- odd extent along every axis,
- dimensionality equal to the network dimensionality.

These rules ensure a well-defined center cell and clear local neighborhoods.

## 1D, 2D, and ND examples

### 1D

```python
feedback = [0.2, 1.0, 0.2]
control = [0.1, 0.8, 0.1]
```

### 2D

```python
feedback = [
    [0.05, 0.20, 0.05],
    [0.20, 1.00, 0.20],
    [0.05, 0.20, 0.05],
]
control = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
]
```

### ND

Use odd-sized stencils whose dimensionality matches the input. In v0.1, ND is best used with SciPy installed for the local aggregation backend.

## Practical heuristics

- Start with small 3-point or 3x3 templates.
- Inspect trajectories, not only the final result.
- Normalize inputs when possible.
- Tune `bias` and `dt` together.
- Use `wrap` for periodic media and `reflect` for finite-domain processing.

## Common mistakes

- Using even-sized templates with no true center.
- Choosing `dt` too large for the feedback strength.
- Expecting PyCNN image-processing defaults to generalize automatically.
- Mixing image I/O directly into the mathematical core.
- Forgetting that CelNN systems are dynamical systems, not one-shot
  linear convolutions.
