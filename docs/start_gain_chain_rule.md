# Starts that depend on the BPM gains: chain rule in the tracking LM fit

`SimulationConfig.starts_follow_gains` with `StartGainMap` (`accelmo/tracking/fitter.py`, `TrackingFitter._solve_lm`).

## Setting

The tracking fit has parameters

- $k$: magnet knobs,
- $g \in \mathbb{R}^{n_g}$: BPM gains, a reading being $(1 + g_b)\,x_b$,

and one start $s_p = (x, p_x, y, p_y)_p$ per tracked particle $p = 1 \dots P$. The loss over the
training workers is

$$
L(k, g, s) = \sum_{p} \sum_{m} w_{pm}\, r_{pm}^2,
\qquad r_{pm} = (1 + g_{b(m)})\, x_{pm}(k, s_p) - d_{pm},
$$

with $d_{pm}$ the measured reading, $w_{pm}$ the normalised inverse variance and $b(m)$ the BPM of
point $m$, scaled by the worker loss scale (`TrackingWorker`, curvature path).

When the starts are reconstructed from the same gained readings (ACD markers rebuilt from the readings
divided by $1 + g$), they are functions $s(g)$ of the gains. The objective is then

$$
\tilde L(k, g) = L\bigl(k, g, s(g)\bigr).
$$

## Linear start map

The initial-conditions callback returns, at the gains $g_\text{ref}$ of the accepted point,

$$
s(g) = s_\text{ref} + S\,(g - g_\text{ref}),
\qquad S = \frac{\partial s}{\partial g}\Big|_{g_\text{ref}} \in \mathbb{R}^{4P \times n_g},
$$

(`StartGainMap.starts` $= s_\text{ref}$, `.jacobian` $= S$ reshaped to $(P, 4, n_g)$,
`.gains` $= g_\text{ref}$). Every evaluation pushes $s(g)$ at the trial gains to the workers
(`StartGainMap.at`), and every LM refresh replaces the map by the callback's new one.

## Gradient

With $\nabla_s L \in \mathbb{R}^{4P}$ the start gradient returned by the workers (`extra["grad"]`),

$$
\nabla_g \tilde L = \nabla_g L + S^\top \nabla_s L,
\qquad \nabla_k \tilde L = \nabla_k L .
$$

## Gauss-Newton Hessian

With residual Jacobians $J_k = \partial r / \partial k$, $J_g = \partial r / \partial g$ and
$J_s = \partial r / \partial s$, the residual Jacobian of $\tilde L$ in $g$ is

$$
\tilde J_g = J_g + J_s S,
$$

and the Gauss-Newton blocks $\tilde H = 2\,\tilde J^\top W \tilde J$ are

$$
\begin{aligned}
\tilde H_{kk} &= H_{kk}, \\
\tilde H_{kg} &= H_{kg} + H_{ks} S, \\
\tilde H_{gg} &= H_{gg} + H_{gs} S + S^\top H_{sg} + S^\top H_{ss} S,
\end{aligned}
$$

with $H_{\cdot\cdot} = 2 J_\cdot^\top W J_\cdot$. The workers return:

| block | worker field | shape |
|---|---|---|
| $H_{ks}$, $H_{gs}$ (rows: knobs, then gains) | `extra["knob_start"]` | $(n_k + n_g,\ 4P)$ |
| $H_{ss}$, block diagonal (one particle's start moves only its own track) | `extra["start_start"]` | $(P, 4, 4)$ |

so with $C = H_{(k,g)s}\,S \in \mathbb{R}^{(n_k + n_g) \times n_g}$ the code adds

$$
\tilde H_{(k,g),g} \mathrel{+}= C, \qquad
\tilde H_{g,(k,g)} \mathrel{+}= C^\top, \qquad
\tilde H_{gg} \mathrel{+}= S^\top H_{ss} S .
$$

The $gg$ block thus receives $C_g + C_g^\top + S^\top H_{ss} S$, which is the three extra terms above.
The gain and knob priors are added after the chain rule, unchanged.

## Particle order

$S$ and $s_\text{ref}$ have one row per particle in `send_init_condition_updates` order: training
workers, then validation workers. The chain rule uses the first $P = \sum$ training particle counts
rows; the validation rows are only pushed.

## Not included

- The dependence of $s$ on the knobs $k$ (the reconstruction model's optics): $\partial s / \partial k$
  is not in the gradient.
- The uncertainty stage (sandwich estimate) treats the starts as fixed.
