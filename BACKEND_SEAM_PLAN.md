# Plan: make the model backend replaceable

Goal: MAD-NG is one implementation of a small backend interface. A second backend (xtrack TPSA first, Cheetah maybe later) is a new subpackage, with no edits to fitters, workers' numerics, optimisers or the protocol. Contract: `docs/replacing_madng.rst`.

Status: see the table at the bottom (kept up to date while implementing).

## Decisions (from the user)

1. Only automatic differentiation computes gradients: MAD-NG TPSA, xtrack TPSA, or Cheetah autograd. No finite-difference backend. Finite differences stay only as an independent check inside tests (`test_madng_xsuite_misalignment.py` keeps its FD reference).
2. The accelerator definitions migrate partly: backend-neutral data (families, errors, misalignments, nonzero rules, BPM patterns, tunes, ACD names) stays in `accelmo.machine`; everything that is Lua code (`_KNOB_PREPARATION`, `knob_expansion_block`, `get_mad_attr_spec` name/sign expressions, LHC bend normalisation, AC-kicker installation) moves to `backend/madng/`.
3. xtrack is the first alternative backend; Cheetah is a maybe.
4. The backend is chosen on the `Accelerator` (`backend="madng"` constructor argument, carried by `copy_with`). No fitter argument.
5. Work on top of the uncommitted WIP (no baseline commit).

## Changes to the original plan (2026-11-22)

* **Element patterns stay in Lua-pattern syntax.** They are a pattern language, not Lua code, and MAD-NG uses them verbatim, so MAD-NG behaviour cannot change. `accelmo.machine.patterns.lua_to_regex` translates them for other backends (tested on every pattern in every `FAMILIES`). Rewriting them as regex and translating back would add a translator with no gain.
* **The `pymadng_utils.accelerators` descriptors stay as base classes.** They are plain Python (energy, beta, sequence name, tunes) and every backend needs them; copying them would duplicate code. The layering rule bans `pymadng` and `pymadng_utils.mad` outside `backend/madng`, not the descriptors.
* **The engine owns the per-worker model build**, so the `Model` protocol is folded into each engine instead of a third object. Only `MachineSetup` (parent process) needs a model without an engine: `backend.open_model(...)`.
* **The engine API mirrors what the workers send today**, one method per Lua block, so step 3/4 is a move of Lua text, not a redesign. Same Lua, same order: numerics are byte-identical.

## Layout

```
accelmo/backend/__init__.py      get_backend(name) -> module with TrackingEngine, ClosedOrbitEngine, open_model, closed_orbit_table
accelmo/backend/protocol.py      Protocols and result dataclasses (no backend import)
accelmo/backend/madng/           everything importing pymadng / pymadng_utils.mad or sending Lua (was machine/mad/)
accelmo/backend/xtrack/          xtrack TPSA engines
accelmo/machine/machine_state.py resolve/merge machine states (pure Python, moved out of machine/mad)
accelmo/machine/patterns.py      lua_to_regex
```

`tests/test_package_layers.py`: only `accelmo/backend/<name>/` imports that backend's library; nothing outside `backend/` imports `accelmo.backend.<name>`.

## Engine interface (`backend/protocol.py`)

Both engines are built inside the worker process from `(WorkerConfig, init_knobs)` and expose `knob_names` (fitted knobs, no `pt`), `set_knobs(knobs)` (ignores names it does not own) and `close()`.

`TrackingEngine(config, plan, init_knobs)`, `plan = TrackingPlan(observables, batch_size, num_batches, tracking_range, n_run_turns, start_on_first_turn, validation, file_index, turn_offset)`:
* `nbpms`, `load_particles(init_coords, init_pts)` (batched lists)
* `set_pts(pt, first_batch, last_batch)`, `set_start_coords(x, px, y, py)`
* `preflight() -> {observed, expected, lost}`
* `track(batch, *, start_derivatives) -> TrackResult(obs, d_dk, d_dc0, n_lost, n_total)`; validation engines return `obs` only.

`ClosedOrbitEngine(config, init_knobs, *, optics_names, orbit_coords, extra_knobs)`:
* `fit_pt`, `read_state(names)`, `assign_state(state)`, `scale_globals([(name, kick, gain, knob)])`
* `set_pt(pt)`, `set_warm_start(key)`
* `solve(observable_names) -> ClosedSolution(names, values, jacobian) | None`
* `plain_knobs()` context manager, `solve_orbit_plain(n_bpms) -> values | None`

Workers keep data prep, weights, loss, gradient, gains, uncertainty and replies.

## Steps

| # | Step | Status |
|---|---|---|
| 1 | `backend/protocol.py`, `backend/__init__.py`, `Accelerator.backend` | |
| 2 | Move `machine/mad/*` to `backend/madng/`; neutral machine-state code to `machine/machine_state.py`; fix all imports (src, tests, scripts) | |
| 3 | `MadngTrackingEngine`; `TrackingWorker` and `AbstractWorker` hold `self.engine`, no `mad` arguments | |
| 4 | `MadngClosedOrbitEngine`; `ClosedTwissWorker`, `ClosedOrbitWorker`, `CalibratedClosedOrbitWorker` use it | |
| 5 | Accelerator Lua hooks to `backend/madng/accelerator_hooks.py`; `lua_to_regex` | |
| 6 | `MachineSetup`, `momentum_reference.closed_orbit_at` through `open_model` / `closed_orbit_table` | |
| 7 | Layer test; MAD-NG suite green on dev3, loss floors unchanged | |
| 8 | xtrack backend: closed orbit (fixed-point TPSA) and tracking (`ParticlesTpsa` per particle); acceptance vs MAD-NG | |

## Risks

* `ParticlesTpsa` is a single map, CPU only: a batch of `P` particles is `P` parametric tracks. Time per batch not measured.
* Which LHC/PSB/SPS elements the xtrack TPSA kernels support is not checked; the xtrack backend refuses knob kinds it cannot map.
* TPSA tracking needs radiation off (fine for the proton machines, not FCC-ee).
* Step 3/4 must keep numerical results identical: same Lua text in the same order, checked with the controller tests' stored values.
