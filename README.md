# accelmo

[![Coverage Status](https://github.com/jgray-19/accelmo/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/accelmo/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/github/jgray-19/accelmo/graph/badge.svg?token=Y1KZACDFPL)](https://codecov.io/github/jgray-19/accelmo)

Estimation of accelerator magnet errors (strengths, misalignments and tilts) from
beam measurements, using gradient-based optimisation of MAD-NG models.

Two families of fit are provided:

| Fit | Data | Entry points |
|---|---|---|
| Tracking | Turn-by-turn BPM data, tracked through the model | `ArcByArcFitter`, `ACDMarkerFitter`, `KickerFitter` |
| Closed orbit | Closed orbit and phase advance, at one or more machine states | `ClosedOrbitFitter`, `CalibratedClosedOrbitFitter` |
| Closed twiss | Closed orbit, phase advance, beta and dispersion | `ClosedOrbitFitter` with `closed_twiss_series` |

Supported machines: LHC, PSB, SPS and FCC.

Full documentation: <https://jgray-19.github.io/accelmo/>

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/jgray-19/accelmo.git
cd accelmo
uv sync
```

Optional dependency groups:

| Extra | Contents |
|---|---|
| `tracking` | omc3, cpymad, pyarrow, psutil (required by the tracking workflows) |
| `monitor` | Panel, for the live fit monitor (`python -m accelmo.monitor`) |
| `test` | pytest, pytest-cov, pytest-xdist |
| `docs` | Sphinx and theme |
| `dev` | ruff, pre-commit |

```bash
uv sync --extra test --extra docs --extra tracking
```

### Companion packages

Companion packages:

| Package | Purpose |
|---|---|
| [pymadng-utils](https://pypi.org/project/pymadng-utils/) | Shared accelerator abstractions, dp/p and pt conversion, knob-file I/O |
| [xtrack_tools](https://github.com/jgray-19/xtrack_tools) | Tracking helpers and dataframe conversion (tests) |

`pymadng-utils` is a normal dependency. `xtrack_tools` is pinned in the `dev` dependency
group, so `uv sync` installs it.

## Usage

A tracking fit is configured with an accelerator and four configuration objects,
then run with `fitter.run()`. It returns a `FitResult`: the fitted knob values,
their 1-sigma uncertainties under the same knob names, and `diagnostics` saying why
and after how many epochs the fit stopped. Optional settings (starting knobs, true
strengths for diagnostics, output and checkpoint settings, callbacks) go in a
`FitterOptions`.

```python
from pathlib import Path

from accelmo.config import OptimiserConfig, TrackingDataConfig, WorkerRuntime
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.machine.accelerators import LHC
from accelmo.tracking import (
    ArcByArcFitter,
    FitterOptions,
    MeasurementConfig,
    MeasurementDetails,
)

accelerator = LHC(beam=1, sequence_file="lhcb1.seq", errors={"quad": {"k1"}})

fitter = ArcByArcFitter(
    accelerator=accelerator,
    solver=OptimiserConfig(
        max_epochs=200,
        warmup_epochs=10,
        warmup_lr_start=1e-6,
        max_lr=1e-4,
        min_lr=1e-6,
    ),
    runtime=WorkerRuntime(num_workers=8),
    tracking_data=TrackingDataConfig(num_batches=4),
    sequence_config=SequenceConfig(magnet_range="$start/$end"),
    measurement_config=MeasurementConfig({Path("measurement.parquet"): MeasurementDetails()}),
    bpm_start_points=["BPM.12R1.B1"],
    bpm_end_points=["BPM.20R1.B1"],
    options=FitterOptions(output_config=OutputConfig(write_run_log=False)),
)
result = fitter.run()
print(result.knobs, result.uncertainties, result.diagnostics.reason)
```

Further examples are in `tests/training/`, which exercise each fitter end to end.

### Tracking modes

| Class | Initial conditions | Tracking |
|---|---|---|
| `ArcByArcFitter` | BPM at the start of each range | Forward and backward over the configured BPM ranges |
| `ACDMarkerFitter` | AC-dipole `before`/`after` markers | Bidirectional, whole ring observed |
| `KickerFitter` | Kicker marker | Single worker, forward only, `turns_per_track` turns |

`KickerFitter` additionally takes the `kicker_name` and `turns_per_track`.
The measurement data must contain `x`, `px`, `y`, `py` at the kicker marker, and the
sequence must include the kicker element.

### Closed-twiss fits

The closed-twiss and closed-orbit fitters live in `accelmo.poco` (Parametric
Optimisation of Closed Orbits) and return the same `FitResult` as the tracking fitters.

`ClosedOrbitFitter(series=closed_twiss_series(measurements, observables))` fits knobs so that
the model's periodic optics match measured closed orbit, beta, phase and dispersion
simultaneously, using a single parametric MAD-NG `twiss`. `measurements` maps the measurement
momentum `pt` to a measured optics dataframe; each becomes one absolute series. The solver is
Levenberg-Marquardt, configured by `LevenbergMarquardtConfig`.

### Closed-orbit fits

`ClosedOrbitFitter` fits knobs to one or more `ClosedOrbitSeries`. Each series
carries a `machine_state`: the MAD-X globals (quadrupole strengths, corrector kicks,
tune knobs) at which its orbits were measured. These are fixed inputs and may differ
between series. Varying the quadrupole strengths between series makes quadrupole
misalignments observable.

Measured orbits are always absolute. A plane in `absolute_planes` is fitted as is; every
other plane is fitted as the change from the fitter's own `machine_state` to the series'
state, the fitter subtracting each measurement's `reference` orbit. `machine_state` accepts a
dictionary, a knobs file or a TFS corrector table; `accelmo.machine.machine_state.merge_machine_states`
combines several. Accepted iterations are recorded in `fitter.history`, and
`fitter.close()` shuts down the MAD-NG process.

See `tests/training/test_closed_orbit_machine_state.py`.

## Tests

```bash
uv run pytest -m "not slow"          # fast suite
uv run pytest -m slow                # convergence and end-to-end tests
uv run pytest --cov=accelmo
```

Markers are listed in `pyproject.toml` and `tests/README.md`.

## Documentation

```bash
uv sync --extra docs
cd docs && uv run make html
```

The output is written to `docs/_build/html/index.html`.

## Related repositories

The measurement and campaign workflows built on this package are maintained
separately: `lhc_measurements`, `psb_md`, `psb_loco`, `lhc_loco` and `fcc_loco`.
