# AffineFlow transition model

AffineFlow encodes the new-command relative degree of smooth torque-driven
mechanics while learning acceleration from endpoint data. It is implemented in
[poc/affine_flow.py](../poc/affine_flow.py). It is a later method version than
the original independent position/velocity power head.

## Learned local law

Let `q,v,u,w ∈ R²`, with joint angles in radians, velocities in rad/s and commands
and activation in N·m. The network learns `a(q,v,effort)=fθ(q,v)+Bθ(q,v)effort`.
The input is `[sin(q1),cos(q1),sin(q2),cos(q2),v1/vscale1,v2/vscale2]`.
A 6→128→128→6 MLP with tanh hidden activations predicts two drift components and
four unconstrained entries of `B`: 18,182 parameters, including biases. The matrix
is neither constrained to be symmetric nor guaranteed to be an inverse mass matrix.

Outputs use training-only velocity scales: `f_i=σv_i z_i/0.04` and
`B_ij=σv_i z_ij/0.08`. Input velocity scales also use training data only.
No interval, new command, activation, true mass matrix or simulator acceleration
enters this coefficient network. Training supervises endpoint position and
velocity with one-step plus four-step closed-loop model-rollout loss.

## Frozen-coefficient propagation

For direct actuation and fixed local `f,B`, the held-command map is:

```text
v′ = v + h f + B h u
q′ = q + h v + ½h² f + B ½h² u
```

For lagged actuation, `w_dot=(u−w)/τ`, with known `τ=0.08 s`, define:

```text
Kw = 1 − exp(−h/τ)
Kv = h − τ Kw
Kq = ½h² − τh + τ² Kw

v′ = v + h f + B [(h−Kv)w + Kv u]
q′ = q + h v + ½h² f + B [(½h²−Kq)w + Kq u]
w′ = w + (u−w)Kw
```

Activation is exact for a held command. The torque contributions are exact when
local coefficients are frozen. Small-interval evaluation uses `expm1` and a
convergent series to avoid subtractive cancellation.

A prediction evaluates `f,B` at the initial state, predicts a midpoint over
`h/2`, evaluates `f,B` there, then applies the full-interval map from the original
state and activation using the midpoint coefficients. The nonlinear mechanical
flow is approximate. Each positive-interval query uses two network evaluations.
Zero time returns the input state without an evaluation.

At fixed initial activation, a new command has leading possible orders
`q:h²,v:h` for direct torque and `q:h³,v:h²` for lagged torque. Structure does
not guarantee a nonzero response, correct finite-horizon gain, good drift,
composition, numerical stability or useful control.

## Competitive NODE implementations

Both controls learn acceleration, enforce `q_dot=v`, receive the same observed
state and propagate the same known activation. On the lagged plant, their
acceleration networks receive activation rather than a direct new-command
shortcut. Both have 17,922 parameters.

- [RK4 NODE](../poc/models/ode.py) uses one classical four-stage Runge–Kutta
  step per requested model interval.
- [Quadrature NODE](../poc/quadrature_node.py) evaluates initial acceleration
  `a0` and predicted-midpoint acceleration `am`. It returns `v′=v+h am` and
  `q′=q+hv+h²(a0/6+am/3)`, preserving the lagged position response a naïve
  midpoint update would miss. It is trained through its own two-evaluation solver.

The final comparison provides the same corpus, objective and nominal-only
optimizer opportunity, with candidate budgets calibrated from timing.
Parameter matching does not imply identical priors or equal fitting accuracy.
[The results](../reports/compute_study_findings.md) retain these distinctions.
