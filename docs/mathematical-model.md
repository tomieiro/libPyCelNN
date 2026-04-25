# Mathematical model and implementation mapping

## Continuous equation

The library uses the canonical CelNN
(Cellular Neural Network) form:

```text
dx/dt = -x + A * y(x) + B * u + z
```

Interpretation:

- `x`: internal state field
- `y(x)`: output field after activation
- `u`: external input field
- `A`: feedback template
- `B`: control template
- `z`: bias term

## Discrete-time approximation

For simulation, the continuous equation is approximated numerically. With explicit Euler:

```text
x_{k+1} = x_k + dt * (-x_k + A * y(x_k) + B * u + z)
```

In the semi-implicit Euler variant implemented in `celnn`, the decay term is treated implicitly:

```text
x_{k+1} = (x_k + dt * (A * y(x_k) + B * u + z)) / (1 + dt)
```

## Local aggregation

`A * y(x)` and `B * u` are implemented as local stencil aggregation over a regular neighborhood. In practical terms:

- templates are odd-shaped arrays,
- the center coefficient corresponds to the current cell,
- neighboring coefficients weight nearby cells.

The implementation uses SciPy acceleration when available and a NumPy pad-and-shift fallback otherwise.

## Template dimensionality

The template dimensionality must match the network dimensionality:

- 1D signal -> 1D template
- 2D image/grid -> 2D template
- ND array -> ND template

In v0.1, robust NumPy fallback support is provided for 1D and 2D. ND aggregation is best used with SciPy installed.

## Boundary handling

Supported boundary modes:

- `constant`
- `wrap`
- `reflect`
- `nearest`
- `mirror`

They determine how neighborhoods are completed at edges before aggregation.

## Output activation

The output is computed by an activation function:

```text
y = f(x)
```

Built-in options include piecewise-linear saturation, identity, `tanh`, sigmoid, sign, and ReLU.

## Convergence

The library reports a simple convergence heuristic based on the maximum absolute state change between the last two stored steps. This is useful for diagnostics, but it is not a formal stability proof.

## Numerical stability warnings

`celnn` emits a basic metadata warning when `dt > 1.0`, because large timesteps can be problematic for explicit schemes. Real stability depends on the full operator, not only on `dt`.

## How Euler works

Explicit Euler is:

- easy to inspect,
- easy to debug,
- often sufficient for small experiments,
- sensitive to step size.

It is the default because it maps transparently onto the CelNN
equation.

## How SciPy `solve_ivp` mode works

When SciPy is available, `solve_ivp` integrates the flattened state vector and reshapes it back to the grid at each RHS evaluation. This is useful for:

- comparing against explicit Euler,
- using a trusted ODE integrator,
- working with ND aggregation through SciPy-backed stencil operations.

## Why `dt` matters

`dt` controls the tradeoff between cost and accuracy:

- smaller `dt` means more steps and usually better fidelity,
- larger `dt` means fewer steps but higher risk of oscillation or divergence.

If a run looks unstable, reduce `dt`, simplify the template, or try the semi-implicit scheme.
