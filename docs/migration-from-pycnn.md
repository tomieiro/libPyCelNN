# Migration from PyCNN

## PyCNN is image-processing oriented

PyCNN is a useful reference implementation for image-processing
examples with CelNN systems. It exposes methods such as:

- `edgeDetection(...)`
- `grayScaleEdgeDetection(...)`
- `cornerDetection(...)`
- `diagonalLineDetection(...)`
- `inversion(...)`
- `optimalEdgeDetection(...)`

## `celnn` is generic

`celnn` generalizes the idea into a reusable framework for:

- 1D signals,
- 2D images,
- 2D grids,
- ND arrays,
- pattern-formation experiments,
- custom local dynamical systems.

## Method-name mapping

### `edgeDetection(...)`

In PyCNN, this is a special image method. In `celnn`, it becomes template-based:

```python
from celnn import CellularNetwork, SimulationConfig
from celnn.templates.image_processing import EDGE_DETECTION

net = CellularNetwork.from_template(
    template=EDGE_DETECTION,
    input=my_array,
    activation="piecewise_linear",
    boundary="reflect",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
```

### `generalTemplates(...)`

PyCNN's generic entry point becomes an explicit network definition:

```python
net = CellularNetwork(
    input=my_array,
    feedback=A,
    control=B,
    bias=Ib,
    initial_state=initial_state,
    boundary="reflect",
)
```

## Image file paths are no longer part of the core

In PyCNN, image-path handling lives near the main processing routines. In `celnn`, image I/O belongs to `celnn.domains.image` and the mathematical core only consumes arrays.

## Mapping of conceptual parameters

- `A` -> `Template.feedback` or `CellularNetwork.feedback`
- `B` -> `Template.control` or `CellularNetwork.control`
- `Ib` -> `Template.bias` or `CellularNetwork.bias`
- initial conditions -> `Template.initial_state` or `CellularNetwork(initial_state=...)`
- time arrays -> `SimulationConfig(t_start=..., t_end=..., dt=...)`

## Practical migration pattern

1. Replace image-path-based method calls with explicit array loading.
2. Replace hardcoded methods with reusable `Template` objects.
3. Create a `CellularNetwork` directly or use `CellularNetwork.from_template(...)`.
4. Move time settings into `SimulationConfig`.
5. Keep image I/O in `celnn.domains.image`.

## PyCNN-inspired templates in `celnn`

Several demonstrative built-ins in `celnn.templates.image_processing` reuse the same kind of examples that appear in PyCNN, with attribution in module metadata. They are provided as migration-friendly starting points, not claims of optimality.
