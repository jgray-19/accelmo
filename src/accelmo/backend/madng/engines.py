"""MAD-NG tracking and closed-orbit engines (see :mod:`accelmo.backend.protocol`).

Each engine owns one MAD-NG process. Derivatives come from the knobs installed as TPSA
parameters on the sequence.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from pymadng_utils.mad.ac_kickers import (
    bind_ac_kicker_volts,
    set_ac_kicker_freqs,
    shift_ac_kicker_phase,
)

from accelmo.backend.madng.accelerator_hooks import track_methods
from accelmo.backend.madng.machine_state import assign_state, read_state
from accelmo.backend.madng.optimising_mad_interface import GradientDescentMadInterface
from accelmo.backend.madng.scripts import (
    CLOSED_TWISS_INIT,
    PYTHON_IN_MAD,
    build_track_speedup_script,
    build_tracking_init_script,
    build_tracking_preflight_script,
    build_tracking_script,
    build_validation_init_script,
    build_validation_script,
    dump_debug_script,
)
from accelmo.backend.protocol import ClosedSolution, TrackResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from pymadng import MAD

    from accelmo.backend.protocol import TrackingPlan
    from accelmo.fitting.worker import WorkerConfig
    from accelmo.machine.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)

#: Observables that are phase-space coordinates, read off the saved map, not optical functions.
ORBIT_COORDS = ("x", "px", "y", "py")


class _MadngSession:
    """The worker's MAD-NG model: sequence, observed BPMs, knobs, cycled start."""

    def __init__(
        self, config: WorkerConfig, init_knobs: Mapping[str, float], logfile: Path | None
    ) -> None:
        iface = GradientDescentMadInterface(
            accelerator=config.accelerator,
            magnet_range=config.magnet_range,
            bpm_range=config.bpm_range,
            **config.interface_options,
            initial_model_values=dict(init_knobs),
            bad_bpms=config.bad_bpms,
            debug=config.debug,
            mad_logfile=logfile,
            py_name=PYTHON_IN_MAD,
            tracking_anchor_mode=config.tracking_anchor_mode,
            tracking_anchor_markers=config.tracking_anchor_sources,
            observed_tracking_anchor_markers=config.observed_tracking_anchor_markers,
        )
        # One contiguous segment from this worker's start; closed-twiss workers do not cycle.
        if config.cycle_target is not None:
            iface.cycle_to_start(config.cycle_target)
        self.iface = iface
        self.mad: MAD = iface.mad
        #: Every knob this process created, ``pt`` last when the energy is fitted.
        self.knob_names: list[str] = list(iface.knob_names)
        self._settable = set(self.knob_names)
        missing = self._settable - set(init_knobs)
        if missing:
            raise ValueError(
                f"{len(missing)} MAD knobs have no initial value, e.g. {sorted(missing)[:5]}"
            )
        mad = self.mad
        mad["knob_names"] = self.knob_names
        # Without a tracking range (kicker mode) MAD tracks the full cycled ring and
        # observes every monitor, including the one wrapping past the start marker.
        self.nbpms = len(iface.all_bpms) if config.tracking_range is None else iface.nbpms
        mad["nbpms"] = self.nbpms
        mad["sdir"] = config.sdir
        mad.import_module("MAD", "damap", "matrix", "vector")
        mad.import_module("MAD.utility", "tblcat")

    def set_knobs(self, knobs: Mapping[str, float], *, plain: bool) -> None:
        """Set this process's knobs; ``plain`` assigns numbers instead of the TPSA constant part."""
        template = "loaded_sequence['{}'] = {:.15e}" if plain else "loaded_sequence['{}']:set0({:.15e})"
        commands = [
            template.format(name, value)
            for name, value in knobs.items()
            if name != "pt" and name in self._settable
        ]
        if commands:
            self.mad.send("\n".join(commands))

    def fitted_knobs_without_pt(self) -> list[str]:
        """``pt`` is applied per particle or pinned on the map, never a sequence knob."""
        names = [name for name in self.mad["knob_names"] if name != "pt"]
        self.mad["knob_names"] = names
        return names

    def create_base_damap(self, knob_order: int = 1) -> None:
        self.mad.send(
            f"da_x0_base = damap{{nv=6, np=#knob_names, "
            f"mo={knob_order}, po={knob_order}, pn=knob_names}}"
        )

    def close(self) -> None:
        self.mad.send("shush()")


class TrackingEngine:
    """MAD-NG ``track`` of one batch of TPSA maps, one map per particle."""

    def __init__(
        self,
        config: WorkerConfig,
        plan: TrackingPlan,
        init_knobs: Mapping[str, float],
        *,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None:
        self.plan = plan
        self.worker_id = worker_id
        self.session = _MadngSession(config, init_knobs, logfile)
        self.knob_names = self.session.knob_names
        self.nbpms = self.session.nbpms
        mad = self.session.mad
        accelerator = config.accelerator
        mad["batch_size"] = plan.batch_size
        mad["num_batches"] = plan.num_batches
        mad["optimise_energy"] = accelerator.optimise_energy
        mad["tracking_range"] = plan.tracking_range
        mad["n_run_turns"] = plan.n_run_turns
        if accelerator.has_ac_kickers:
            set_ac_kicker_freqs(mad, accelerator.ac_kicker_freqs(plan.file_index))
            volt_knobs = accelerator.ac_kicker_volt_knobs(plan.file_index)
            if volt_knobs is not None:
                bind_ac_kicker_volts(mad, volt_knobs)
            shift_ac_kicker_phase(mad, plan.turn_offset)
        self._install_knob_parameters(accelerator)
        self._build_scripts(config)
        self._save_start = False

    def _install_knob_parameters(self, accelerator: Accelerator) -> None:
        """Knobs as TPSA parameters; a validation engine tracks numbers only."""
        mad = self.session.mad
        if self.plan.validation:
            mad.send("da_x0_base = damap{nv=6, np=0, mo=1, po=1}")
            return
        self.session.fitted_knobs_without_pt()
        self.session.create_base_damap(knob_order=1)
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    loaded_sequence[param] = loaded_sequence[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")
        mad.send(build_track_speedup_script(track_methods(accelerator)))

    def _build_scripts(self, config: WorkerConfig) -> None:
        observables = self.plan.observables
        if self.plan.validation:
            self._init_script = build_validation_init_script(observables)
            self._track_script = build_validation_script(observables)
            self._start_track_script = self._track_script
            return
        self._init_script = build_tracking_init_script(
            observables, start_on_first_turn=self.plan.start_on_first_turn
        )
        self._track_script = build_tracking_script(observables)
        self._start_track_script = build_tracking_script(
            observables, include_start_derivatives=True
        )
        for name, text in (
            ("run_track_init", self._init_script),
            ("run_track", self._track_script),
            ("run_track_uncertainty", self._start_track_script),
        ):
            dump_debug_script(
                name, text, debug=config.debug, mad_logfile=config.mad_logfile, worker_id=self.worker_id
            )

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        self.session.set_knobs(knobs, plain=self.plan.validation)

    def load_particles(
        self, init_coords: list[list[list[float]]], init_pts: list[list[float]]
    ) -> None:
        """One DA map per particle in each batch, at its start coordinates."""
        mad = self.session.mad
        mad.send("""
init_coords = python:recv()
init_pts = python:recv()
""")
        mad.send(init_coords).send(init_pts)
        mad.send("""
da_x0_c = table.new(num_batches, 0)
for i=1,num_batches do
    da_x0_c[i] = table.new(batch_size, 0)
    for j=1,batch_size do
        da_x0_c[i][j] = da_x0_base:copy()
        da_x0_c[i][j]:set0(init_coords[i][j])
    end
end
""")
        mad.send(self._init_script)

    def set_pts(self, pt: float, first_batch: int, last_batch: int) -> None:
        self.session.mad.send(f"""
for b = {first_batch + 1}, {last_batch + 1} do
    for i = 1, batch_size do
        da_x0_c[b][i].pt:set0({pt:.15e} + init_pts[b][i])
    end
end
""")

    def set_start_coords(
        self, x: np.ndarray, px: np.ndarray, y: np.ndarray, py: np.ndarray
    ) -> None:
        """Constant parts of x, px, y, py of every map; ``t``, ``pt`` and the DA coefficients stay."""
        mad = self.session.mad
        # N x 1 column matrices (binary protocol); a single MAD matrix index is linear.
        mad.send("""
new_x  = python:recv()
new_px = python:recv()
new_y  = python:recv()
new_py = python:recv()

local particle = 0
for batch=1,num_batches do
    for j=1,#da_x0_c[batch] do
        particle = particle + 1
        da_x0_c[batch][j].x:set0(new_x[particle])
        da_x0_c[batch][j].px:set0(new_px[particle])
        da_x0_c[batch][j].y:set0(new_y[particle])
        da_x0_c[batch][j].py:set0(new_py[particle])
    end
end
""")
        for values in (x, px, y, py):
            mad.send(np.asarray(values, dtype=np.float64).reshape(-1, 1))

    def preflight(self) -> dict[str, int | bool]:
        mad = self.session.mad
        mad.send(build_tracking_preflight_script())
        return mad.recv()

    def _set_save_start(self, enabled: bool) -> None:
        if enabled != self._save_start:
            self.session.mad.send(f"save_start_derivatives = {'true' if enabled else 'false'}")
            self._save_start = enabled

    def track(self, batch: int, *, start_derivatives: bool) -> TrackResult:
        mad = self.session.mad
        observables = self.plan.observables
        mad.send(f"batch = {batch + 1}")
        if self.plan.validation:
            mad.send(self._track_script)
            return TrackResult({o: np.asarray(mad.recv()).squeeze(-1) for o in observables})
        self._set_save_start(start_derivatives)
        mad.send(self._start_track_script if start_derivatives else self._track_script)
        loss_info: dict = mad.recv()
        obs = {o: np.asarray(mad.recv()).squeeze(-1) for o in observables}
        # MAD stores (points, knobs) rows; the result is (particles, knobs, points).
        d_dk = {o: np.stack(mad.recv(), axis=0).transpose(0, 2, 1) for o in observables}
        d_dc0 = (
            {o: np.stack(mad.recv(), axis=0) for o in observables} if start_derivatives else None
        )
        return TrackResult(obs, d_dk, d_dc0, int(loss_info["n_lost"]), int(loss_info["n_total"]))

    def close(self) -> None:
        self.session.close()


def _strip_comment_lines(text: str) -> str:
    """Remove full-line comments and blank lines before sending to MAD-NG."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--") or stripped.startswith("!"):
            continue
        kept.append(line)
    return "\n".join(kept)


def read_orbit_only(mad: MAD, n_coords: int, n_knobs: int):
    """BPM names, orbit values and Jacobian after ``compute_closed_orbit``."""
    mad.send("send_orbit_only()")
    names = list(mad.recv())
    values = np.empty((n_coords, len(names)))
    jacobian = np.empty((n_coords, len(names), n_knobs))
    columns = max(n_knobs, 1)  # MAD keeps one all-zero column when there are no knobs
    for i in range(n_coords):
        values[i] = np.asarray(mad.recv(), dtype=float).ravel()
        jacobian[i] = np.asarray(mad.recv(), dtype=float).reshape(len(names), columns)[:, :n_knobs]
    return names, values, jacobian


def _knob_monomial(index: int, n_knobs: int) -> str:
    """Parameter-monomial suffix selecting d/d(knob ``index``) in a trkopt name.

    ``gphys.nf_pk`` reads the ``_`` separator as the six phase-space slots, so only the
    parameter part belongs in the suffix.
    """
    return "0" * index + "1" + "0" * (n_knobs - index - 1)


class ClosedOrbitEngine:
    """Parametric closed orbit (``cofind`` + one ``track``) or closed twiss in MAD-NG.

    Optical functions and their knob derivatives come from twiss's ``trkopt`` columns; the
    orbit coordinates from the saved map (``run_closed_twiss_init.mad``).
    """

    def __init__(
        self,
        config: WorkerConfig,
        init_knobs: Mapping[str, float],
        *,
        optics_names: Sequence[str],
        orbit_coords: Sequence[str],
        extra_knobs: Sequence[str] | None = None,
        worker_id: int,
        logfile: Path | None = None,
    ) -> None:
        del worker_id
        self.session = _MadngSession(config, init_knobs, logfile)
        mad = self.session.mad
        self.optics_names = list(optics_names)
        self.orbit_coords = list(orbit_coords)
        if extra_knobs is None:
            self.fit_pt = "pt" in self.session.knob_names
        else:
            # Calibrated fit: zero-valued gain knobs after the magnet knobs; pt is never fitted.
            if extra_knobs:
                mad.send("\n".join(f"loaded_sequence['{knob}'] = 0" for knob in extra_knobs))
            mad["knob_names"] = [name for name in mad["knob_names"] if name != "pt"] + list(
                extra_knobs
            )
            self.fit_pt = False
        #: Parametric knobs in Jacobian column order (magnet knobs, then extra knobs).
        self.knob_names = self.session.fitted_knobs_without_pt()
        self.n_knobs = len(self.knob_names)
        self._value_columns = [f"{name}_" for name in self.optics_names]
        self._derivative_columns = [
            f"{name}_{_knob_monomial(i, self.n_knobs)}"
            for name in self.optics_names
            for i in range(self.n_knobs)
        ]
        mad["optics_columns"] = self._value_columns + self._derivative_columns
        mad["orbit_coords"] = self.orbit_coords
        mad.send(_strip_comment_lines(CLOSED_TWISS_INIT.read_text()))
        self._solver = "compute_closed_twiss()" if self.optics_names else "compute_closed_orbit()"

    def set_knobs(self, knobs: Mapping[str, float]) -> None:
        self.session.set_knobs(knobs, plain=False)

    def read_state(self, names: list[str]) -> dict[str, float]:
        return read_state(self.session.mad, PYTHON_IN_MAD, names)

    def assign_state(self, state: Mapping[str, float]) -> None:
        assign_state(self.session.mad, state)

    def scale_globals(self, scaled: Sequence[tuple[str, float, float, str]]) -> None:
        """Deferred ``kick * (gain + knob)``: follows the knobs when they are made plain."""
        commands = [
            f"MADX['{name}'] = \\-> {kick:.15e} * ({gain:.15e} + loaded_sequence['{knob}'])"
            for name, kick, gain, knob in scaled
        ]
        if commands:
            self.session.mad.send("\n".join(commands))

    def set_pt(self, pt: float) -> None:
        self.session.mad.send(f"x0map.pt:set0({pt:.15e})")

    def set_warm_start(self, key: str) -> None:
        self.session.mad.send(f"co_key = '{key}'")

    def _found(self, command: str, what: str) -> bool:
        mad = self.session.mad
        mad.send(command)
        try:
            found = mad.recv()
        except RuntimeError as exc:  # MAD raised: an unstable trial point
            LOGGER.warning("MAD error in the %s (%s)", what, exc)
            return False
        return bool(found)

    def solve(self, observables: Sequence[str]) -> ClosedSolution | None:
        if not self._found(self._solver, "closed-orbit solve"):
            return None
        mad = self.session.mad
        n_knobs = self.n_knobs
        if not self.optics_names:
            return ClosedSolution(*read_orbit_only(mad, len(self.orbit_coords), n_knobs))

        columns = ["name", *self.orbit_coords, *self._value_columns, *self._derivative_columns]
        frame = mad.closed_tbl.to_df(columns=columns)
        names = list(frame["name"])
        n_bpms = len(names)
        n_optics = len(self.optics_names)
        optics_values = frame[self._value_columns].to_numpy(dtype=float).T
        optics_jacobian = (
            frame[self._derivative_columns]
            .to_numpy(dtype=float)
            .reshape(n_bpms, n_optics, n_knobs)
            .transpose(1, 0, 2)
        )
        orbit_values = frame[list(self.orbit_coords)].to_numpy(dtype=float).T
        orbit_jacobian = np.empty((len(self.orbit_coords), n_bpms, n_knobs))
        if self.orbit_coords:
            mad.send("send_orbit_jacobian()")
            for i in range(len(self.orbit_coords)):
                orbit_jacobian[i] = np.asarray(mad.recv(), dtype=float).reshape(n_bpms, n_knobs)

        values = np.empty((len(observables), n_bpms))
        jacobian = np.empty((len(observables), n_bpms, n_knobs))
        orbit_iter, optics_iter = iter(range(len(self.orbit_coords))), iter(range(n_optics))
        for i, name in enumerate(observables):
            if name in ORBIT_COORDS:
                source = next(orbit_iter)
                values[i], jacobian[i] = orbit_values[source], orbit_jacobian[source]
            else:
                source = next(optics_iter)
                values[i], jacobian[i] = optics_values[source], optics_jacobian[source]
        return ClosedSolution(names, values, jacobian)

    @contextmanager
    def plain_knobs(self) -> Iterator[None]:
        self.session.mad.send("knobs_to_plain()")
        try:
            yield
        finally:
            self.session.mad.send("knobs_to_param()")

    def solve_orbit_plain(self, n_bpms: int) -> np.ndarray | None:
        if self.optics_names:
            raise NotImplementedError("loss-only evaluation supports orbit observables only")
        if not self._found("compute_orbit_plain()", "loss-only orbit"):
            return None
        mad = self.session.mad
        values = np.empty((len(self.orbit_coords), n_bpms))
        for index in range(len(self.orbit_coords)):
            values[index] = np.asarray(mad.recv(), dtype=float).ravel()
        return values

    def close(self) -> None:
        self.session.close()


def open_model(
    accelerator: Accelerator,
    *,
    magnet_range: str,
    machine_state: object = None,
    b2_errors: Path | None = None,
    bad_bpms: list[str] | None = None,
    debug: bool = False,
    mad_logfile: Path | None = None,
    anchor_mode: str | None = None,
    anchor_markers: Sequence[str] = (),
) -> GradientDescentMadInterface:
    """Parent-side model: knob discovery and BPM lists."""
    return GradientDescentMadInterface(
        accelerator=accelerator,
        magnet_range=magnet_range,
        machine_state=machine_state,
        b2_errors=b2_errors,
        bad_bpms=bad_bpms,
        debug=debug,
        mad_logfile=mad_logfile,
        tracking_anchor_mode=anchor_mode,
        tracking_anchor_markers=list(anchor_markers),
    )


def closed_orbit_table(
    accelerator: Accelerator,
    magnet_strengths: Mapping[str, float] | None,
    pt: float,
    machine_state: object,
    columns: Sequence[str],
):
    """Coupled twiss columns at every observed element, ``pt`` pinned on the initial condition."""
    iface = GradientDescentMadInterface(accelerator, machine_state=machine_state)
    try:
        if magnet_strengths:
            iface.update_knob_values(dict(magnet_strengths))
        iface.mad.send(
            f"moref = twiss{{sequence=loaded_sequence, observe=1, "
            f"X0={{pt={float(pt):.15e}}}, coupling=true}}"
        )
        frame = iface.mad.moref.to_df(columns=["name", *columns])
    finally:
        iface.close()
    return frame.set_index("name")[list(columns)]
