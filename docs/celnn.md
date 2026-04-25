# CelNN: concept, theory, and study notes

## 1. What CelNN systems are

`CelNN` is the convention used by this library for Cellular Neural
Networks. This project standardizes on `CelNN` to avoid confusion with
Convolutional Neural Networks.

CelNN systems are arrays of locally coupled dynamical cells. Each cell
has an internal state, an output, and a local neighborhood through
which it interacts with nearby cells and external inputs.

The key idea is local interaction with global consequences: the update law is local, but the observed pattern can be global and emergent.

## 2. Why they are not Convolutional Neural Networks

CelNN systems are **not** Convolutional Neural Networks.

The name overlap is historical. Cellular Neural Networks were
introduced in the late 1980s as analog-style nonlinear dynamical
systems with local coupling. Convolutional Neural Networks became
dominant much later in machine learning as feed-forward or recurrent
deep-learning architectures based on learned convolutions and
end-to-end optimization.

In this library:

- `CelNN` means **Cellular Neural Network**.
- `CelNN` does **not** mean Convolutional Neural Network.
- `CelNN` means a locally connected cellular dynamical system.

## 3. Historical context: Chua and Yang, 1988

Leon O. Chua and Lin Yang introduced Cellular Neural Networks in 1988
as large-scale nonlinear systems built from identical local cells
arranged on regular lattices. Their work framed these systems as
analog and massively parallel information-processing systems that are
especially natural for local interactions and spatially distributed
dynamics.

## 4. Core intuition

Think of a grid of cells. Each cell:

- stores an internal state `x`,
- emits an output `y(x)`,
- receives an external input `u`,
- is influenced only by nearby cells through local templates.

Even though each cell only sees a small neighborhood, repeated evolution can produce edge maps, smoothed signals, thresholded patterns, traveling fronts, diffusion-like behavior, and other collective effects.

## 5. Mathematical model

A common continuous-time model is:

```text
dx/dt = -x + A * y(x) + B * u + z
```

where:

- `x` is the internal state,
- `y(x)` is the output or activation,
- `u` is the external input,
- `A` is the feedback template,
- `B` is the control template,
- `z` or `Ib` is the bias,
- `*` denotes local neighborhood aggregation.

## 6. State, output, and input

The state `x` is the internal dynamical quantity. The output `y(x)` is usually a bounded nonlinear function of the state. The input `u` is an externally supplied field such as:

- a 1D signal,
- a 2D image,
- a 2D or ND grid,
- a structured initial excitation field.

## 7. Feedback template `A`

The feedback template determines how neighboring outputs influence the current cell. It defines the local recurrent coupling. If the center coefficient is strong and positive, the current cell tends to reinforce its own output. If surrounding coefficients are smoothing-like or inhibitory, the dynamics can diffuse, sharpen, or compete.

## 8. Control template `B`

The control template weights the external input `u`. It determines how the static or time-independent driving input affects the local dynamics. In many image-processing examples, `B` is where edge-like or detection-like stencils appear.

## 9. Bias `z` or `Ib`

The bias shifts thresholds and can change the qualitative behavior of a network. A negative bias may suppress weak activity. A positive bias may encourage activation. In practice, bias is one of the fastest levers for changing how selective a template behaves.

## 10. Activation and output function

The output function `y(x)` maps internal state to observable output.
Classical CelNN literature often uses a piecewise-linear saturating
nonlinearity. Other practical choices include `tanh`, identity, sign,
sigmoid, or ReLU-like functions depending on the experiment.

## 11. Boundary conditions

Every local stencil needs a rule at the edges:

- `constant`: pad with a fixed value,
- `wrap`: periodic domain,
- `reflect`: reflect without repeating the edge value,
- `nearest`: extend the edge value,
- `mirror`: symmetric reflection.

Boundary conditions matter because they change how the system behaves near borders.

## 12. Continuous-time dynamics

The continuous equation describes a coupled nonlinear ODE system. For a
grid with many cells, the full system is high-dimensional, but still
structured because the coupling is local. This locality makes CelNN
systems closely related to stencil-based dynamical systems.

## 13. Discrete numerical simulation

A Python library does not simulate the system in exact continuous time. It approximates the ODE numerically:

- explicit Euler,
- semi-implicit Euler for the linear decay term,
- `solve_ivp` when SciPy is available.

The smaller the timestep `dt`, the closer the discrete approximation is to the underlying continuous model, though computational cost increases.

## 14. Relationship to cellular automata

Cellular automata use local rules over neighboring cells too, but
their updates are typically discrete in time and state. CelNN systems
are closer to continuous-state and often continuous-time dynamical
systems. They can approximate thresholding and logic-like behavior,
but they also support smooth analog dynamics.

## 15. Relationship to PDE-like stencil computations

CelNN systems are strongly related to finite-difference and stencil
computations. A local template is mathematically close to a small
stencil operator. Diffusion-like, Laplacian-like, and reaction-like
behavior can often be expressed through suitable templates plus
nonlinear output functions.

## 16. Relationship to signal processing

In 1D, a CelNN can act like a nonlinear local filter:

- smoothing,
- thresholding,
- denoising,
- edge-like or transition detection,
- nonlinear dynamical filtering.

## 17. Relationship to image processing

Image processing is one important application, not the whole story.
Historically, many published CelNN templates target:

- edge detection,
- corner detection,
- line detection,
- inversion,
- segmentation-like thresholding.

But a general-purpose library should expose the underlying framework rather than hide it behind image-specific method names.

## 18. Relationship to pattern formation

Because CelNN systems couple local recurrence and nonlinearity, they
naturally support pattern-formation experiments. Small changes in
templates, bias, initial conditions, or boundary handling can produce
dramatically different spatial organization.

## 19. Stability intuition

The `-x` term is a local decay term. It acts like damping. Stability, however, still depends on:

- feedback strength,
- nonlinearity,
- timestep size,
- solver,
- input scale,
- boundary conditions.

Large positive feedback and large `dt` can produce oscillations or numerical instability.

## 20. Template design intuition

Useful mental patterns:

- center-dominant positive weights often preserve or reinforce structure,
- neighboring positive weights often smooth or diffuse,
- mixed positive and negative weights often detect contrasts or transitions,
- bias shifts the effective threshold,
- output saturation limits runaway growth but does not guarantee stability.

## 21. Common applications

- image and signal filtering,
- local feature detection,
- thresholding and logic-like dynamics,
- spatially distributed dynamical systems,
- diffusion-like or reaction-like pattern experiments,
- educational demonstrations of local nonlinear dynamics.

## 22. Limitations

- Template design is usually manual or heuristic.
- Stability is not automatic.
- Interpretation depends on scaling and activation choice.
- Generic graph neighborhoods are possible in principle, but this library focuses on regular arrays in v0.1.
- The pure NumPy fallback is robust for 1D and 2D; higher-dimensional aggregation is best used with SciPy installed.

## 23. Recommended study path

1. Learn the continuous equation and the role of `A`, `B`, `x`, `y(x)`, `u`, and `z`.
2. Simulate 1D signals with simple 3-point templates.
3. Move to 2D image-like and grid-like examples.
4. Compare activations and boundary modes.
5. Study stability by varying `dt` and feedback strength.
6. Design your own templates and inspect trajectories, not just final states.

## 24. Glossary

- **Cell**: one local dynamical unit in the field.
- **State**: internal variable `x`.
- **Output**: transformed observable `y(x)`.
- **Template**: local stencil of coefficients.
- **Feedback template**: recurrent template `A`.
- **Control template**: input template `B`.
- **Bias**: offset term `z` or `Ib`.
- **Boundary condition**: edge-handling rule.
- **Trajectory**: stored sequence of states during simulation.

## 25. References

1. L. O. Chua and L. Yang, “Cellular Neural Networks: Theory,” *IEEE Transactions on Circuits and Systems*, 1988.
2. L. O. Chua and L. Yang, “Cellular Neural Networks: Applications,” *IEEE Transactions on Circuits and Systems*, 1988.
3. PyCNN repository by Ankit Aggarwal, MIT License: <https://github.com/ankitaggarwal011/PyCNN>
4. Tutorial literature on image processing and pattern formation in Cellular Neural Networks.

## Closing note

CelNN systems are **not** Convolutional Neural Networks.

They are locally connected nonlinear dynamical systems over cellular
fields. That distinction matters for both the mathematics and the
software design.
