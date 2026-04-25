# Template creation guide

This guide explains how to create new CelNN templates for new tasks.
It complements [template-design.md](template-design.md): that document
covers the design principles, while this one is a practical workflow
for building a template for a specific problem.

## 1. Start from the CelNN equation

The working model in `celnn` is:

```text
dx/dt = -x + A * y(x) + B * u + z
```

When you design a template, you are not choosing only a matrix. You are
choosing a local dynamical system made of:

- `A`: feedback template
- `B`: control template
- `z`: bias
- `y(x)`: activation
- `dt`: numerical timestep
- `boundary`: edge handling
- `initial_state`: optional starting condition

So the practical question is not only “what goes in the matrix?” but
also “which part of the behavior should come from feedback, input,
thresholding, boundaries, and time evolution?”

## 2. What each piece does

### `A`: feedback template

`A` acts on the current output `y(x)`. It controls internal recurrence.

Use `A` when the task depends on:

- local memory
- self-reinforcement
- competition between neighbors
- diffusion-like evolution
- pattern formation
- thresholded collective behavior

Typical interpretation:

- strong positive center: preserve or reinforce the current cell
- positive neighbors: smoothing, spreading, diffusion
- negative neighbors: inhibition, contrast, competition
- asymmetric neighbors: directional effects

### `B`: control template

`B` acts on the external input `u`. It controls how the input enters the
system.

Use `B` when the task depends on:

- filtering the input directly
- extracting local contrast from the input
- detecting transitions, peaks, or edges
- making the final behavior more input-driven than self-driven

Typical interpretation:

- center-dominant `B`: pass-through or weighted local averaging
- zero-sum `B`: contrast or edge-like response
- directional `B`: oriented detection

### `bias`

`bias` shifts the effective threshold.

Use bias to:

- suppress weak responses
- encourage activation
- change the operating point of the network
- make a detector more selective or less selective

Typical behavior:

- negative bias: harder to activate
- positive bias: easier to activate
- zero bias: neutral baseline

### `activation`

The activation converts internal state to output.

Use it to control:

- saturation
- threshold sharpness
- smoothness
- whether output should stay bounded

### `dt`

`dt` is not part of the template, but it changes the behavior
substantially. It controls how aggressively the continuous dynamics are
approximated in discrete time.

Use it to control:

- numerical stability
- speed of convergence
- smoothness of the trajectory

### `boundary`

`boundary` defines what happens near the edges of the domain.

Use it to control:

- whether borders behave like hard walls
- whether the domain wraps around
- whether edges are reflected or repeated

### `initial_state`

`initial_state` matters mostly when the behavior is dynamical rather
than purely input-driven.

Use it when:

- the task depends on transient behavior
- you want symmetry breaking
- you want pattern formation from random initial conditions

## 3. How to interpret each matrix entry

In a 2D `3x3` template:

```text
w11 w12 w13
w21 w22 w23
w31 w32 w33
```

`w22` is the weight of the current cell. The other entries correspond to
neighbors at those relative offsets.

Interpretation:

- `w22`: self-coupling
- `w12`: top neighbor
- `w21`: left neighbor
- `w23`: right neighbor
- `w32`: bottom neighbor
- diagonals: corner interactions

In 1D:

```text
[w_left, w_center, w_right]
```

The same idea applies.

Practical rules:

- larger magnitude means stronger influence
- positive means reinforcement
- negative means inhibition
- symmetric weights mean no preferred direction
- asymmetric weights mean direction-sensitive behavior

## 4. First design question: input-driven or dynamics-driven?

This is the most important design decision.

### Mostly input-driven tasks

Examples:

- smoothing a noisy signal
- sharpening an image
- edge detection
- local peak detection
- local transition detection

Recommended starting point:

- put most of the logic in `B`
- keep `A` simple, often identity-like or weak
- use small `dt`
- use bounded activation if you want stable output ranges

### Mostly dynamics-driven tasks

Examples:

- diffusion-like evolution
- relaxation processes
- spatial competition
- pattern formation
- self-organized thresholding

Recommended starting point:

- put most of the logic in `A`
- make `B` weak or zero
- choose `initial_state` carefully
- tune `dt` conservatively

## 5. Choosing `bias`, `dt`, activation, and boundary

These choices are often as important as the template itself.

## 5.1 Choosing `bias`

Use `bias` based on the role you want:

- `bias < 0`: use when the detector is firing too easily
- `bias = 0`: use when you want a neutral starting point
- `bias > 0`: use when activity is too weak or too easily suppressed

Good situations for negative bias:

- edge detectors with too many false positives
- thresholding tasks where only strong responses should survive
- pattern tasks where the system grows too aggressively

Good situations for positive bias:

- weak signals
- templates that damp too much
- cases where you want broad activation regions

Tuning rule:

1. Start with `0.0`.
2. If output is too active everywhere, move negative.
3. If output is too quiet everywhere, move positive.

## 5.2 Choosing `dt`

Use `dt` based on stability and how fine you want the evolution to be.

Recommended starting ranges:

- `0.01` to `0.05`: careful and stable default
- `0.05` to `0.1`: practical default for many small experiments
- `> 0.1`: use only if the system remains stable

When to use smaller `dt`:

- strong positive feedback
- highly nonlinear activations
- pattern-formation tasks
- oscillatory or unstable trajectories

When a larger `dt` may be acceptable:

- weak feedback
- mostly input-driven filters
- short experiments where behavior is already smooth

Warning signs that `dt` is too large:

- oscillation that should not be there
- exploding values before saturation
- sharp frame-to-frame jumps
- strong sensitivity to tiny parameter changes

## 5.3 Choosing the activation

### `piecewise_linear`

Use when:

- you want the classical CelNN style
- you want bounded output in `[-1, 1]`
- you want a good default for most experiments

Best for:

- general-purpose templates
- image-like and grid-like tasks
- tasks inspired by classical CelNN literature

### `saturated_linear`

Use when:

- you want clipping but simple behavior
- you want output bounded in `[-1, 1]`
- you want less sharp corners than `sign_activation`

Best for:

- stable bounded simulations
- diffusion-like and pattern-like experiments

### `identity`

Use when:

- you want to study the linear or near-linear effect directly
- you do not want saturation to hide what the stencil is doing

Best for:

- debugging
- linear filter intuition
- benchmarking template structure

Be careful:

- it can become unstable more easily

### `tanh_activation`

Use when:

- you want smooth saturation
- you want a softer nonlinear response than
  `piecewise_linear`

Best for:

- smooth signal filtering
- soft pattern dynamics
- cases where hard saturation feels too abrupt

### `sigmoid_activation`

Use when:

- output should live near `[0, 1]`
- the task is naturally one-sided rather than signed

Best for:

- probabilistic-looking outputs
- occupancy-like maps
- positive-only response tasks

### `sign_activation`

Use when:

- you want hard thresholding
- the task is closer to binary decisions

Best for:

- logic-like behavior
- hard segmentation
- thresholding demos

Be careful:

- it can make tuning rough and discontinuous

### `relu_activation`

Use when:

- negative values should be suppressed entirely
- only positive activation matters

Best for:

- one-sided detection tasks
- positive-only amplification

## 5.4 Choosing `boundary`

### `constant`

Use when:

- outside the domain should be treated as fixed background
- borders should behave like hard edges

Best for:

- images with known background level
- zero-padding style experiments

### `reflect`

Use when:

- the domain is finite but you do not want hard-edge artifacts
- border behavior should resemble a mirrored continuation

Best for:

- image filtering
- many finite-domain signal tasks

This is usually the safest practical default.

### `nearest`

Use when:

- edge values should be repeated outward

Best for:

- piecewise-constant boundary assumptions
- plateau-like signals

### `mirror`

Use when:

- you want symmetric mirrored continuation
- you need a reflection mode different from `reflect`

Best for:

- careful experiments comparing edge semantics

### `wrap`

Use when:

- the domain is periodic
- the left edge should connect to the right edge
- the top edge should connect to the bottom edge

Best for:

- periodic media
- toroidal grids
- cyclic signals
- pattern formation on periodic domains

## 6. A practical workflow for a new task

Use this sequence.

### Step 1: define the task clearly

Ask:

- Do I want smoothing, contrast, thresholding, directionality, or
  self-organization?
- Is the task about the input itself or about the evolution?
- Do I care more about the final state or the trajectory?

### Step 2: choose the smallest reasonable neighborhood

Start small:

- 1D tasks: `3`
- 2D tasks: `3x3`
- only move to `5x5` or larger if `3x3` cannot express the behavior

### Step 3: decide where the behavior lives

- If the task is mostly a local filter, start with `B`
- If the task is mostly recurrent dynamics, start with `A`

### Step 4: choose the center weight

The center usually determines how strongly a cell keeps its own value.

Rules of thumb:

- large positive center: preserve local state
- small center: let neighbors dominate
- negative center: strong local suppression, use with care

### Step 5: choose neighbor weights

Ask what neighbors should do:

- support the center
- suppress the center
- detect contrast with the center
- prefer one direction

### Step 6: choose `bias`

Use `bias` to move the threshold after the stencil is approximately
right.

### Step 7: choose activation

Choose based on whether you want:

- soft bounded behavior
- hard thresholding
- near-linear analysis

### Step 8: choose boundary

Choose based on whether your data is:

- finite and non-periodic
- periodic
- background-padded

### Step 9: choose `dt`

Start conservative. A good default is:

```python
SimulationConfig(t_end=5.0, dt=0.05)
```

### Step 10: test on synthetic cases first

Do not start with complicated real data. Start with:

- a constant field
- a single impulse
- a step edge
- a checkerboard
- a noisy sine wave

These reveal what the template is really doing.

## 7. Common task patterns

## 7.1 Smoothing or denoising

Goal:

- reduce local noise
- preserve broad structure

Start with:

```python
feedback = [
    [0.05, 0.10, 0.05],
    [0.10, 0.40, 0.10],
    [0.05, 0.10, 0.05],
]
control = [
    [0.0, 0.0, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 0.0, 0.0],
]
```

Recommended settings:

- activation: `tanh_activation` or `piecewise_linear`
- bias: `0.0`
- boundary: `reflect`
- `dt`: `0.01` to `0.05`

## 7.2 Edge or contrast detection

Goal:

- respond to local differences
- suppress flat regions

Start with:

```python
feedback = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
]
control = [
    [-1.0, -1.0, -1.0],
    [-1.0,  8.0, -1.0],
    [-1.0, -1.0, -1.0],
]
```

Recommended settings:

- activation: `piecewise_linear`
- bias: slightly negative
- boundary: `reflect`
- `dt`: `0.05` to `0.1`

## 7.3 Directional detection

Goal:

- respond more strongly to one direction than another

Strategy:

- break symmetry in `B`
- keep `A` simple at first

Example:

```python
control = [
    [-1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, -1.0],
]
```

Recommended settings:

- activation: `piecewise_linear`
- bias: moderate negative if the detector is too sensitive
- boundary: `reflect`

## 7.4 Thresholding or logic-like behavior

Goal:

- create hard or soft binary-like regions

Strategy:

- moderate `A`
- use `bias` to set selectivity
- use `sign_activation` for hard decisions or
  `piecewise_linear` for soft ones

Recommended settings:

- activation: `sign_activation` or `piecewise_linear`
- bias: usually nonzero
- boundary: `reflect` or `constant`
- `dt`: small

## 7.5 Diffusion-like behavior

Goal:

- spread local energy smoothly
- relax sharp local variations

Strategy:

- use positive neighbor weights in `A`
- keep `B` small or zero

Recommended settings:

- activation: `saturated_linear` or `tanh_activation`
- bias: `0.0`
- boundary: `wrap` for periodic media, `reflect` otherwise
- `dt`: conservative, often `0.01` to `0.05`

## 7.6 Pattern formation

Goal:

- generate or stabilize nontrivial spatial structure

Strategy:

- use `A` as the main design lever
- start from random `initial_state`
- often use `wrap` boundaries
- keep `dt` small

Recommended settings:

- activation: `saturated_linear` or `tanh_activation`
- bias: small negative or near zero
- boundary: `wrap`
- `dt`: `0.01` to `0.05`

## 7.7 Peak detection in 1D

Goal:

- identify local maxima or strong transitions in a signal

Strategy:

- use a center-positive, neighbors-negative `B`
- keep `A` weak at first

Example:

```python
feedback = [0.0, 0.5, 0.0]
control = [-1.0, 2.0, -1.0]
```

Recommended settings:

- activation: `identity` while debugging, then
  `piecewise_linear`
- bias: slightly negative if too many peaks survive
- boundary: `reflect`
- `dt`: `0.01` to `0.05`

## 8. How to write your own template in `celnn`

```python
from celnn.templates import Template

my_template = Template(
    name="my_task_template",
    feedback=[
        [0.05, 0.20, 0.05],
        [0.20, 1.00, 0.20],
        [0.05, 0.20, 0.05],
    ],
    control=[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    bias=-0.1,
    description="Starter template for a custom local task.",
    tags=["custom", "demo"],
)
```

Then use it:

```python
from celnn import CellularNetwork, SimulationConfig

net = CellularNetwork.from_template(
    template=my_template,
    input=my_array,
    activation="piecewise_linear",
    boundary="reflect",
)

result = net.run(SimulationConfig(t_end=5.0, dt=0.05))
```

## 9. Debugging checklist

If the template does not behave as expected:

- test with `identity` activation first
- reduce `dt`
- set `bias = 0.0` and tune it later
- simplify to `3` or `3x3`
- inspect `trajectory_output`, not only final output
- test with synthetic inputs
- check boundary effects explicitly
- check whether the task logic should live in `A` or `B`

## 10. Quick summary

Use this condensed decision map:

- Want a local filter from the input:
  start with `B`
- Want recurrent evolution:
  start with `A`
- Too much activation:
  lower `bias`
- Too little activation:
  raise `bias`
- Unstable simulation:
  reduce `dt`
- Need bounded smooth output:
  use `tanh_activation` or `saturated_linear`
- Need classical CelNN behavior:
  use `piecewise_linear`
- Need hard thresholding:
  use `sign_activation`
- Need periodic domain:
  use `wrap`
- Need practical finite-domain behavior:
  use `reflect`

The most effective workflow is iterative: start small, inspect the
trajectory, adjust one parameter at a time, and only then move to more
complex templates.
