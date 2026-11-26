# /// script
# requires-python = ">=3.12"
# dependencies = ["accelmo[tracking]>=0.0.2", "xtrack>=0.118.6", "xtrack-tools"]
# [tool.uv.sources]
# xtrack-tools = { git = "https://github.com/jgray-19/xtrack_tools", rev = "0e6779d35546db12182a99d25c1cba60d395c90e" }
# ///
"""LOCO-style POCO fit of the LHC beam-1 main quadrupoles with MAD-NG and xsuite.

The fit frees the 392 arc main quadrupole strengths (``MQ.*.dk1l``). The data are orbit
responses: for each selected horizontal and vertical corrector (150 per plane by default),
the closed-orbit change at every BPM when that corrector alone is kicked
(``ClosedOrbitSeries`` with the corrector kick as machine state, orbit relative to the unkicked
reference). Correctors are inputs, never fitted.

The correctors of each plane are chosen from the MAD-NG twiss of the nominal lattice: one per
slot of ring phase, alternating between two phases pi/2 apart mod pi, among the correctors with
at least the lower-quartile beta of their plane. The script prints the largest phase gap and
the quadrature imbalance of every plane's choice.

The data are generated once, by xtrack alone, from random quadrupole errors of 1e-4 relative to
each ``k1l``, with the tunes then rematched through ``dqx.b1_op`` / ``dqy.b1_op`` (the matched
trims become the machine state of both fits), and saved to ``truth_data.npz`` in the output
directory (a saved file is reused if its settings match; delete it to regenerate). The generating
line is the most accurate one: full dipole edges, ``bend-kick-bend`` bends with one kick (as
MAD-NG's bend body; four kicks change the response by less than 1e-10 relative), exact drifts,
``mat-kick-mat-exact`` quadrupoles and ``drift-kick-drift-exact`` sextupoles/octupoles
(Yoshida 6), sliced until the response changed by 1e-11 relative against four times finer
slicing. Both fits then use their own, faster default models.

```bash
uv run examples/benchmark_madng_vs_xsuite_lhc_loco.py
uv run examples/benchmark_madng_vs_xsuite_lhc_loco.py --backends madng --workers 32
```

Writes ``truth_data.npz``, ``results.json`` and ``correctors.csv`` to the output directory.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xtrack as xt

from accelmo.backend.madng import MadInterface
from accelmo.backend.xtrack.model import XtrackModel
from accelmo.config import WorkerRuntime
from accelmo.fitting.config import OutputConfig, SequenceConfig
from accelmo.logs import configure_logging
from accelmo.machine.accelerators import LHC
from accelmo.poco import ClosedOrbitFitter, ClosedOrbitMeasurement, ClosedOrbitSeries
from accelmo.poco.fitter import LevenbergMarquardtConfig

REPO = Path(__file__).resolve().parents[1]
SEQUENCE = REPO / "tests" / "data" / "sequences" / "lhcb1.seq"
BACKENDS = ("madng", "xtrack")
PLANES = {"H": ("hkicker", "mu1", "beta11"), "V": ("vkicker", "mu2", "beta22")}
TUNE_KNOBS = ("dqx.b1_op", "dqy.b1_op")  # operational tune trims, rematched after the errors
CORRECTORS_PER_PLANE = 150
KICK = 1e-5  # rad
REL_ERROR = 1e-4  # random quadrupole error, relative to k1l
BPM_ERROR = 1e-8  # m, weight only: the data are noise-free
MAX_ITERATIONS = 15
NUM_WORKERS = 16
SEED = 1
TRUTH_QUAD_KICKS, TRUTH_BEND_KICKS, TRUTH_SEXTUPOLE_KICKS = 112, 1, 128
MIN_BETA_QUANTILE = 0.25


def accelerator(backend: str) -> LHC:
    return LHC(beam=1, sequence_file=SEQUENCE, errors={"quad": {"k1"}}, backend=backend)


def nominal_correctors() -> tuple[pd.DataFrame, dict[str, float]]:
    """Every corrector of beam 1 (kind, phases [2pi], betas) and the ring tune per plane."""
    interface = MadInterface(accelerator("madng"), create_knobs=True)
    try:
        twiss = interface.run_twiss(observe=0, deltap=0.0, coupling=False, method=6)
    finally:
        interface.close()
    kinds = {kind for kind, _, _ in PLANES.values()}
    table = twiss[twiss["kind"].isin(kinds)][["kind", "s", "mu1", "mu2", "beta11", "beta22"]]
    table.index = table.index.astype(str).str.upper()
    return table, {"H": float(twiss.headers["q1"]), "V": float(twiss.headers["q2"])}


def select_correctors(table: pd.DataFrame, plane: str, count: int, total: float) -> pd.DataFrame:
    """``count`` correctors of ``plane``, one per slot of ring phase, alternating in quadrature.

    Slot ``k`` targets ring phase ``(k + 0.5) * total / count`` and phase ``0`` (even ``k``) or
    ``pi/2`` (odd ``k``) mod pi, so neighbouring slots sample the orbit's sine and cosine
    components. The unused corrector with at least the lower-quartile beta of the plane that
    minimises the ring-phase distance (in slots) plus the circular mod-pi distance (in
    quarter turns) takes each slot.
    """
    kind, mu, beta = PLANES[plane]
    plane_table = table[table["kind"] == kind]
    candidates = plane_table[plane_table[beta] >= plane_table[beta].quantile(MIN_BETA_QUANTILE)]
    phases = candidates[mu].to_numpy()
    mod_pi = phases % 0.5
    chosen: list[int] = []
    for slot in range(count):
        wanted = 0.25 * (slot % 2)
        circular = np.abs(mod_pi - wanted)
        circular = np.minimum(circular, 0.5 - circular)
        cost = np.abs(phases - (slot + 0.5) * total / count) / (total / count) + circular / 0.25
        chosen.append(int(next(i for i in np.argsort(cost) if i not in chosen)))
    return candidates.iloc[sorted(chosen)]


def phase_statistics(selected: pd.DataFrame, plane: str, total: float) -> dict[str, float]:
    mu = selected[PLANES[plane][1]].to_numpy()
    gaps = np.diff(np.r_[mu, mu[0] + total])
    return {
        "max_phase_gap_2pi": float(gaps.max()),
        "uniform_phase_gap_2pi": total / len(mu),
        "mod_pi_quadrature_imbalance": float(abs(np.exp(4j * np.pi * mu).mean())),
        "min_beta": float(selected[PLANES[plane][2]].min()),
    }


def accurate_truth_model() -> XtrackModel:
    """The xtrack LHC line with converged slicing, for generating the data."""
    model = XtrackModel(accelerator("xtrack"))
    line = model.line
    line.configure_drift_model(model="exact")
    line.configure_quadrupole_model(
        model="mat-kick-mat-exact", integrator="yoshida-6", num_multipole_kicks=TRUTH_QUAD_KICKS
    )
    line.configure_bend_model(
        core="bend-kick-bend", edge="full", num_multipole_kicks=TRUTH_BEND_KICKS
    )
    for configure in (line.configure_sextupole_model, line.configure_octupole_model):
        configure(
            model="drift-kick-drift-exact",
            integrator="yoshida-6",
            num_multipole_kicks=TRUTH_SEXTUPOLE_KICKS,
        )
    return model


def data_settings(correctors: list[str]) -> str:
    """Everything the saved data depends on besides the lattice files."""
    return json.dumps(
        {
            "correctors": correctors,
            "kick": KICK,
            "rel_error": REL_ERROR,
            "seed": SEED,
            "truth_kicks": [TRUTH_QUAD_KICKS, TRUTH_BEND_KICKS, TRUTH_SEXTUPOLE_KICKS],
            "tune_knobs": TUNE_KNOBS,
        }
    )


def generate_truth(path: Path, correctors: list[str]) -> None:
    """Random quadrupole errors, tunes rematched, and the orbit unkicked and with each kick."""
    model = accurate_truth_model()
    line = model.line
    nominal = line.twiss(method="4d", only_orbit=False)
    rng = np.random.default_rng(SEED)
    knob_names = list(model.knob_names)
    k1l = np.array(
        [
            float(line[name[: -len(".dk1l")].lower()].k1)
            * float(line[name[: -len(".dk1l")].lower()].length)
            for name in knob_names
        ]
    )
    truth = REL_ERROR * k1l * rng.normal(size=len(knob_names))
    model.set_knob_values(dict(zip(knob_names, truth.tolist(), strict=True)))
    line.match(
        vary=[xt.Vary(knob, step=1e-6) for knob in TUNE_KNOBS],
        targets=[xt.Target("qx", nominal.qx, tol=1e-9), xt.Target("qy", nominal.qy, tol=1e-9)],
        method="4d",
    )
    tune_state = [float(line.vars[knob]._value) for knob in TUNE_KNOBS]
    rows = [name.lower() for name in model.all_bpms]

    def orbit() -> np.ndarray:
        twiss = line.twiss(method="4d", only_orbit=True)
        return np.array([twiss.rows[rows][plane] for plane in ("x", "y")])

    orbits = [orbit()]
    for name in correctors:
        model.set_vars({f"{name}.kick": KICK})
        orbits.append(orbit())
        model.set_vars({f"{name}.kick": 0.0})
    np.savez(
        path,
        bpms=np.array(model.all_bpms),
        correctors=np.array(correctors),
        orbits=np.array(orbits),
        knob_names=np.array(knob_names),
        truth=truth,
        k1l=k1l,
        tune_state=np.array(tune_state),
        settings=data_settings(correctors),
    )


def load_truth(path: Path, correctors: list[str]):
    """Saved data: orbits per series (``""`` is unkicked), truth, prior widths, tune-knob state."""
    data = np.load(path)
    if str(data["settings"]) != data_settings(correctors):
        raise ValueError(f"{path} was made with other settings; delete it to regenerate")
    bpms = data["bpms"].tolist()
    orbits = {
        name: pd.DataFrame(
            {"X": orbit[0], "Y": orbit[1], "ERRX": BPM_ERROR, "ERRY": BPM_ERROR}, index=bpms
        )
        for name, orbit in zip(["", *correctors], data["orbits"], strict=True)
    }
    knob_names = data["knob_names"].tolist()
    truth = dict(zip(knob_names, data["truth"].tolist(), strict=True))
    sigmas = dict(zip(knob_names, (REL_ERROR * np.abs(data["k1l"])).tolist(), strict=True))
    tune_state = dict(zip(TUNE_KNOBS, data["tune_state"].tolist(), strict=True))
    return orbits, truth, sigmas, tune_state


def fit(
    backend: str,
    orbits: dict[str, pd.DataFrame],
    correctors: list[str],
    sigmas: dict[str, float],
    tune_state: dict[str, float],
    workers: int,
    max_iterations: int,
) -> tuple[dict[str, float], dict]:
    series = [
        ClosedOrbitSeries(
            measurements=(ClosedOrbitMeasurement(orbits[name], reference=orbits[""]),),
            machine_state={f"{name}.kick": KICK},
            label=name,
        )
        for name in correctors
    ]
    fitter = ClosedOrbitFitter(
        accelerator=accelerator(backend),
        sequence_config=SequenceConfig(magnet_range="$start/$end"),
        series=series,
        lm_config=LevenbergMarquardtConfig(max_iterations=max_iterations),
        knob_sigmas=sigmas,
        machine_state=tune_state,
        output_config=OutputConfig(write_run_log=False, include_uncertainty=False),
        runtime=WorkerRuntime(num_workers=workers),
    )
    start = time.perf_counter()
    try:
        result = fitter.run()
    finally:
        fitter.close()
    elapsed = time.perf_counter() - start
    diagnostics = result.diagnostics
    return result.knobs, {
        "fit_s": elapsed,
        "iterations": diagnostics.iterations,
        "best_loss": diagnostics.best_loss,
        "converged": diagnostics.converged,
        "reason": diagnostics.reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_madng_vs_xsuite_lhc"))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--correctors-per-plane", type=int, default=CORRECTORS_PER_PLANE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "logs" / "run.log")

    table, tunes = nominal_correctors()
    selected = {
        plane: select_correctors(table, plane, args.correctors_per_plane, tunes[plane])
        for plane in PLANES
    }
    statistics = {plane: phase_statistics(selected[plane], plane, tunes[plane]) for plane in PLANES}
    correctors = [name for plane in PLANES for name in selected[plane].index]
    pd.concat(selected.values()).to_csv(args.output_dir / "correctors.csv")
    print(json.dumps(statistics, indent=2))

    data_path = args.output_dir / "truth_data.npz"
    if not data_path.exists():
        generate_truth(data_path, correctors)
    orbits, truth, sigmas, tune_state = load_truth(data_path, correctors)
    knob_names = list(truth)
    truth_vector = np.array([truth[name] for name in knob_names])

    results = {}
    for backend in args.backends:
        fitted, timing = fit(
            backend, orbits, correctors, sigmas, tune_state, args.workers, args.max_iterations
        )
        error = np.array([fitted[name] for name in knob_names]) - truth_vector
        results[backend] = {
            **timing,
            "rel_error": float(np.linalg.norm(error) / np.linalg.norm(truth_vector)),
            "rms_error": float(np.sqrt(np.mean(error**2))),
            "max_abs_error": float(np.abs(error).max()),
        }
        print(backend, json.dumps(results[backend]), flush=True)
    settings = {
        "knobs": len(knob_names),
        "correctors": {plane: len(selected[plane]) for plane in PLANES},
        "bpms": len(orbits[""]),
        "kick_rad": KICK,
        "rel_error": REL_ERROR,
        "tune_state": tune_state,
        "workers": args.workers,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps({"settings": settings, "phase": statistics, "results": results}, indent=2)
    )
    print(json.dumps(settings))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
