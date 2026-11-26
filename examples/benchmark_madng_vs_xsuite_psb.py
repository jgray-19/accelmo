# /// script
# requires-python = ">=3.12"
# dependencies = ["accelmo[tracking]>=0.0.2", "xtrack>=0.118.6", "xtrack-tools"]
# [tool.uv.sources]
# xtrack-tools = { git = "https://github.com/jgray-19/xtrack_tools", rev = "0e6779d35546db12182a99d25c1cba60d395c90e" }
# ///
"""Speed of MAD-NG and xsuite (xtrack TPSA) for the PSB fits: POCO and the DLMN methods.

One evaluation is what a worker computes per optimiser step: the model values and their
full Jacobian with respect to the 48 quadrupole ``k1`` knobs of PSB ring 3.

- ``poco``: closed orbit ``x, y`` at every BPM (POCO, closed-orbit fit).
- ``dlmn_single_turn``: a batch of reconstructed initial conditions, each tracked once
  through the ring from the first to the last BPM (single-turn method).
- ``dlmn_kicker_multi_turn``: a few kicked initial conditions tracked for many turns from
  the kicker (kicker method, multi-turn). The AC-dipole multi-turn method needs the ACD
  amplitude and phase as TPSA parameters, which xsuite elements do not offer
  (``docs/backends.rst``), so it is not benchmarked.

Both backends run in this process, single-threaded, through
:mod:`accelmo.backend.protocol`, so the numbers include the Python side of each engine.
The script also reports the largest difference between the two codes' results.

Run from the repository root (needs ``tests/data/acc-models-psb``):

```bash
uv run examples/benchmark_madng_vs_xsuite_psb.py
uv run examples/benchmark_madng_vs_xsuite_psb.py --methods poco --profile
uv run --with-editable /path/to/xtrack examples/benchmark_madng_vs_xsuite_psb.py
```

Writes ``timings.csv`` and ``timings.json`` to the output directory and prints a table.
``--profile`` also prints the cProfile of one xtrack evaluation per method, sorted by
cumulative time. ``--with-editable`` benchmarks a local xtrack checkout.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-threaded timings

import argparse
import cProfile
import json
import logging
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omc3.model_creator import create_instance_and_model
from pymadng_utils.madx.make_sequence import make_madx_sequence

from accelmo.backend import get_backend
from accelmo.backend.protocol import TrackGeometry, TrackRequest
from accelmo.fitting.worker import WorkerConfig
from accelmo.machine.accelerators import PSB

REPO = Path(__file__).resolve().parents[1]
ACC_MODELS = REPO / "tests" / "data" / "acc-models-psb"
BACKENDS = ("madng", "xtrack")
METHODS = ("poco", "dlmn_single_turn", "dlmn_kicker_multi_turn")
ERRORS = {"quad": {"k1"}}
KNOB_SPREAD = 2e-3  # integrated dk1l [1/m], uniform +-
CORRECTOR_STATE = {"kbr3dhz2l4": 2e-4, "kbr3dvt3l4": -1.5e-4}  # a non-zero closed orbit
SINGLE_TURN_PARTICLES = 100
MULTI_TURN_PARTICLES = 2
MULTI_TURN_TURNS = 2000
KICKER = "BE3.KFA14L1"  # PSB ring-3 injection/extraction kicker; tracking starts here
START_AMPLITUDE = 1e-3  # m, transverse start coordinates
REPEATS = 5
SEED = 1
PROFILE_LINES = 30

LOGGER = logging.getLogger(__name__)


def psb_sequence(output_dir: Path) -> Path:
    """PSB ring 3 sequence from the omc3 nominal model of the test fixtures (built once)."""
    model_dir = output_dir / "psb_model"
    sequence = model_dir / "psb3_saved.seq"
    if not sequence.exists():
        create_instance_and_model(
            outputdir=model_dir,
            accel="psbooster",
            type="nominal",
            nat_tunes=[0.17, 0.225],
            dpp=0.0,
            fetch="path",
            path=ACC_MODELS,
            scenario="lhc_indiv",
            year="2026",
            cycle_point="1_flat_bottom",
            str_file="psb_fb_lhcindiv.str",
            ring=3,
            driven_excitation="acd",
            drv_tunes=[0.162, 0.232],
            list_choices=False,
            show_help=False,
            logfile=None,
        )
        make_madx_sequence(model_dir)
    return sequence


def accelerator(sequence: Path, backend: str) -> PSB:
    return PSB(ring=3, sequence_file=sequence, errors=ERRORS, backend=backend)


def knob_values(sequence: Path) -> tuple[dict[str, float], list[str]]:
    model = get_backend("madng").open_model(
        accelerator(sequence, "madng"), magnet_range="$start/$end"
    )
    names, bpms = list(model.knob_names), list(model.all_bpms)
    model.close()
    rng = np.random.default_rng(SEED)
    return {name: float(rng.uniform(-KNOB_SPREAD, KNOB_SPREAD)) for name in names}, bpms


def start_coordinates(n: int) -> list[list[list[float]]]:
    rng = np.random.default_rng(SEED)
    coords = np.zeros((n, 6))
    coords[:, :4] = rng.uniform(-START_AMPLITUDE, START_AMPLITUDE, (n, 4)) * [1, 0.1, 1, 0.1]
    return [coords.tolist()]


def timed(
    function, backend: str, profile: bool, repeats: int = REPEATS
) -> tuple[np.ndarray, object]:
    """Wall times of ``repeats`` calls after one warm-up call, and the last result.

    With ``profile`` the xtrack backend gets one extra profiled call (cumulative time).
    """
    result = function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    if profile and backend == "xtrack":
        profiler = cProfile.Profile()
        profiler.runcall(function)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(PROFILE_LINES)
    return np.array(times), result


def poco(sequence: Path, backend: str, knobs: dict[str, float], profile: bool):
    config = WorkerConfig(
        accelerator=accelerator(sequence, backend),
        magnet_range="$start/$end",
        machine_state=CORRECTOR_STATE,
    )
    start = time.perf_counter()
    engine = get_backend(backend).ClosedOrbitEngine(
        config, knobs, optics_names=[], orbit_coords=["x", "y"], worker_id=0
    )
    engine.set_pt(0.0)
    engine.set_knobs(knobs)
    setup = time.perf_counter() - start
    times, solution = timed(lambda: engine.solve(["x", "y"]), backend, profile)
    order = [engine.knob_names.index(name) for name in sorted(knobs)]
    engine.close()
    return setup, times, solution.values, solution.jacobian[:, :, order]


def dlmn(
    sequence: Path,
    backend: str,
    knobs: dict[str, float],
    bpms: list[str],
    profile: bool,
    *,
    multi_turn: bool,
):
    n_particles = MULTI_TURN_PARTICLES if multi_turn else SINGLE_TURN_PARTICLES
    config = WorkerConfig(accelerator=accelerator(sequence, backend), magnet_range="$start/$end")
    plan = TrackRequest(
        geometry=TrackGeometry(
            start_bpm=bpms[0],
            end_bpm=bpms[-1],
            initial_condition_marker=KICKER if multi_turn else None,
            n_turns=MULTI_TURN_TURNS if multi_turn else 1,
        ),
        observables=("x", "px", "y", "py"),
        batch_size=n_particles,
        num_batches=1,
        validation=False,
        file_index=0,
        turn_offset=0,
    )
    start = time.perf_counter()
    engine = get_backend(backend).TrackingEngine(config, plan, knobs, worker_id=0)
    engine.load_particles(start_coordinates(n_particles), [[0.0] * n_particles])
    engine.set_knobs(knobs)
    engine.set_pts(0.0, 0, 0)
    setup = time.perf_counter() - start
    times, result = timed(lambda: engine.track(0, start_derivatives=False), backend, profile)
    order = [engine.knob_names.index(name) for name in sorted(knobs)]
    engine.close()
    values = np.stack([result.obs[o] for o in ("x", "y")])
    jacobian = np.stack([result.d_dk[o][:, order] for o in ("x", "y")])
    return setup, times, values, jacobian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_madng_vs_xsuite"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--profile", action="store_true", help="cProfile one xtrack evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = psb_sequence(args.output_dir)
    knobs, bpms = knob_values(sequence)

    methods = {
        "poco": lambda backend: poco(sequence, backend, knobs, args.profile),
        "dlmn_single_turn": lambda backend: dlmn(
            sequence, backend, knobs, bpms, args.profile, multi_turn=False
        ),
        "dlmn_kicker_multi_turn": lambda backend: dlmn(
            sequence, backend, knobs, bpms, args.profile, multi_turn=True
        ),
    }
    rows = []
    for method in args.methods:
        run = methods[method]
        results = {backend: run(backend) for backend in BACKENDS}
        (_, _, v_mad, j_mad), (_, _, v_xs, j_xs) = results["madng"], results["xtrack"]
        for backend, (setup, times, _, _) in results.items():
            rows.append(
                {
                    "method": method,
                    "backend": backend,
                    "setup_s": setup,
                    "eval_mean_s": float(times.mean()),
                    "eval_std_s": float(times.std(ddof=1)),
                    "max_value_diff": float(np.abs(v_mad - v_xs).max()),
                    "max_jacobian_diff": float(np.abs(j_mad - j_xs).max()),
                    "max_jacobian": float(np.abs(j_mad).max()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "timings.csv", index=False)
    settings = {
        "knobs": len(knobs),
        "bpms": len(bpms),
        "single_turn_particles": SINGLE_TURN_PARTICLES,
        "multi_turn_particles": MULTI_TURN_PARTICLES,
        "multi_turn_turns": MULTI_TURN_TURNS,
        "repeats": REPEATS,
    }
    (args.output_dir / "timings.json").write_text(
        json.dumps({"settings": settings, "timings": rows}, indent=2)
    )
    print(json.dumps(settings))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
