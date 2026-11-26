User Guide
==========

Tracking fits
-------------

A tracking fit recovers knob strengths by tracking measured initial conditions
through the MAD-NG model and comparing the result with the measured turn-by-turn
data. A fit is defined by an accelerator, four configuration objects and an optional
:class:`~accelmo.tracking.fitter.FitterOptions`. ``fitter.run()`` returns a
:class:`~accelmo.fitting.results.FitResult`: the fitted knobs, their 1-sigma
uncertainties under the same knob names, and ``diagnostics`` (why and after how many
epochs the fit stopped).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Object
     - Role
   * - :class:`~accelmo.machine.accelerators.base.Accelerator` subclass
     - Machine definition: sequence file, kinetic energy, BPM pattern and the knob
       families to fit (``errors``, ``misalignments``).
   * - ``solver``
     - :class:`~accelmo.config.OptimiserConfig` (``adam`` or ``lbfgs``: epoch count,
       learning-rate schedule, stopping rule) or a Levenberg-Marquardt config from
       :func:`~accelmo.tracking.fitter.tracking_lm_config`.
   * - :class:`~accelmo.config.WorkerRuntime`
     - Worker count, worker log level and BLAS threads.
   * - :class:`~accelmo.config.TrackingDataConfig`
     - Batch count, data and validation fractions, momenta in the loss, track length.
   * - :class:`~accelmo.fitting.config.SequenceConfig`
     - Magnet range exposed to MAD-NG and BPMs to exclude.
   * - :class:`~accelmo.tracking.config.models.MeasurementConfig`
     - Measurement files, each with a :class:`~accelmo.tracking.config.models.MeasurementDetails`
       (MAD interface options, momentum offset, first BPM).
   * - :class:`~accelmo.tracking.fitter.FitterOptions` (optional)
     - Starting knobs, true strengths, output and checkpoint settings, callbacks, and the
       :class:`~accelmo.config.TrackingParameters` (BPM gains, starts, prior widths).

Fitters
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Fitter
     - Initial conditions
     - Tracking
   * - :class:`~accelmo.tracking.fitter.ArcByArcFitter`
     - BPM at the start of each range
     - Forward and backward over the ``bpm_start_points`` x ``bpm_end_points`` ranges.
       Set ``acd_excited=True`` for AC-dipole data.
   * - :class:`~accelmo.tracking.fitter.ACDMarkerFitter`
     - AC-dipole ``before``/``after`` markers
     - Bidirectional; the whole ring is observed.
   * - :class:`~accelmo.tracking.fitter.KickerFitter`
     - Kicker marker
     - One worker, forward only, ``turns_per_track`` turns.
   * - :class:`~accelmo.tracking.fitter.ACDSinusoidFitter`
     - Before-marker, every ``turns_per_track`` turns
     - Back-to-back ``turns_per_track``-turn tracks through the model AC kickers.

All fit families, the knob families of each machine and the solvers are listed in
:doc:`fit_families`.

``KickerFitter`` also takes the ``kicker_name`` and ``turns_per_track``.
The measurement data must contain ``x``, ``px``, ``y`` and ``py`` at the kicker
marker, and the sequence must contain the kicker element.

Example
~~~~~~~

.. code-block:: python

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
       measurement_config=MeasurementConfig(
           {Path("measurement.parquet"): MeasurementDetails()}
       ),
       bpm_start_points=["BPM.12R1.B1"],
       bpm_end_points=["BPM.20R1.B1"],
       options=FitterOptions(output_config=OutputConfig(write_run_log=False)),
   )
   result = fitter.run()
   print(result.knobs, result.uncertainties, result.diagnostics.reason)

FitterOptions
~~~~~~~~~~~~~

``initial_knob_strengths``
    Starting values, in optimisation space. Knobs left out keep the model value.
``true_strengths``
    Reference values (file or dictionary) used to report the error of the fit.
``debug``
    Run MAD-NG in debug mode.
``output_config``
    :class:`~accelmo.fitting.config.OutputConfig`: run log, TensorBoard logging,
    uncertainty estimation and log files.
``checkpoint_config``
    :class:`~accelmo.tracking.config.models.CheckpointConfig`: periodic checkpoints and restart.
``initial_conditions_callback``
    Epoch-end hook that returns refreshed start coordinates for every worker, or
    ``None`` to keep them.
``loss_callback``
    Called each epoch with the loss values.

Validation
~~~~~~~~~~

``TrackingDataConfig.validation_fraction`` (default 0.1) holds out a disjoint set of turns
per file. These are never used for training, so the validation loss is out-of-sample.
``data_fraction`` is applied to the remaining training turns.

Closed-twiss fits
-----------------

The closed-twiss and closed-orbit fitters live in :mod:`accelmo.poco` and return the
same :class:`~accelmo.fitting.results.FitResult` as the tracking fitters.

:class:`~accelmo.poco.closed_orbit.ClosedOrbitFitter` with
``series=closed_twiss_series(measurements, observables)``
(:func:`~accelmo.poco.closed_orbit.closed_twiss_series`) fits knobs so that the periodic model
optics match a measured closed twiss. Closed orbit, beta, phase and
dispersion are all obtained from one parametric MAD-NG ``twiss``, so they are fitted
simultaneously and no starting point is taken from the measurement. The solver is
Levenberg-Marquardt, configured by
:class:`~accelmo.optimisers.levenberg_marquardt.LevenbergMarquardtConfig`.

``measurements`` maps each measurement's ``pt`` to a measured optics dataframe; each is one
absolute series. Parameters are
weighted by the inverse measurement variance; ``use_errors=False`` normalises every
observable family identically instead. ``knob_sigmas`` adds a Gaussian prior of absolute widths.

Closed-orbit fits
-----------------

:class:`~accelmo.poco.closed_orbit.ClosedOrbitFitter` fits knobs to one or
more :class:`~accelmo.poco.closed_orbit.ClosedOrbitSeries`.

``machine_state``
    MAD-X globals (quadrupole strengths, corrector kicks, tune knobs) at which a
    series was measured. They are fixed inputs and may differ between series.
    Varying the quadrupole strengths between series makes quadrupole misalignments
    observable. Accepted forms are a dictionary, a knobs file or a TFS corrector
    table; :func:`accelmo.machine.machine_state.merge_machine_states` combines several. Setting
    ``machine_state`` on the fitter provides a default inherited by every series.
``absolute_planes``
    Planes fitted as absolute orbits. The remaining planes are fitted as the change
    from the fitter's own ``machine_state`` to the series' state: the fitter subtracts
    each measurement's ``reference`` orbit (errors added in quadrature), which is
    required for those planes. Measured orbits are always passed in absolute.
``observables``
    Fitted quantities: ``x``, ``y``, and the phase advances ``mu1``, ``mu2``
    (``mu1``/``mu2`` are set per series).

Accepted iterations are stored in ``fitter.history``. Call ``fitter.close()`` to shut
down the MAD-NG process once the fit is complete.

:class:`~accelmo.poco.calibrated.CalibratedClosedOrbitFitter` extends this with BPM gains and corrector kick
calibrations as additional parameters, with Gaussian priors ``sigma_bpm`` and
``sigma_corrector``. Every ``k_<corrector>`` kick a series' ``machine_state`` sets gets a gain; a
series may set several or none (an off-momentum orbit). ``fit_corrector_gains=False`` fits the BPM
gains alone, which is the mode for absolute orbits.

Degeneracy analysis
-------------------

:mod:`accelmo.analysis.degeneracy_checker` evaluates the eigenspectrum of the
Gauss-Newton normal matrix at the initial knobs. Near-zero eigenvalues identify knob
combinations that the data cannot constrain, so the problem can be regularised or the
knob set reduced before running the fit. Use
:meth:`~accelmo.tracking.fitter.TrackingFitter.check_degeneracy` on a tracking fitter.

Terminal dashboard and logs
---------------------------

A tracking fit draws a live terminal dashboard (:class:`~accelmo.dashboard.FitDashboard`):
a strip of stages (data, model, workers, fit, uncertainty), progress with ETA, the loss
history, the best epoch so far and the last warnings. It does nothing when the output is
not a terminal; the run log has the numbers then.

Call :func:`accelmo.logs.configure_logging` once at the top of a script to print WARNING
and above to the terminal and, with ``file=``, write INFO and above to a file. The
dashboard hides the console handler while it is on screen and shows warnings in its own
strip.

Each worker process writes every record at ``WorkerRuntime.logging_level`` (default
DEBUG) to ``workers/python_worker_<id>.log`` in the run log directory. Only WARNING
and above are relayed to the main process, tagged with the worker name, so they reach the
dashboard and the main log. A worker with its own ``python_logfile`` keeps it.

.. note::

   Workers are forked processes, and the warning relay reaches them only by fork (Linux, the
   default start method up to Python 3.13). The ``spawn`` and ``forkserver`` start methods, the
   default on macOS, Windows and Python 3.14, are not supported: a spawned worker would write its
   own log file but its warnings would not reach the main process or the dashboard.

Monitoring a fit
----------------

Every fit writes a run log under ``OutputConfig.run_log_root`` (switch off with
``OutputConfig(write_run_log=False)``). Serve live plots of running and finished fits
with:

.. code-block:: bash

   uv sync --extra monitor
   python -m accelmo.monitor RUN_LOG_ROOT --spec my_package.monitor:SPEC

``accelmo-monitor`` is the same command as ``python -m accelmo.monitor``. Then open
``http://localhost:5006``. On a remote machine, forward the port with
``ssh -L 5006:localhost:5006 host``. Without ``--spec`` only the generic panels are
shown. See :mod:`accelmo.monitor` and :mod:`accelmo.runlog`.

Testing
-------

.. code-block:: bash

   pytest -m "not slow"     # fast suite
   pytest -m slow           # convergence and end-to-end tests
   pytest --cov=accelmo

Markers are listed in ``tests/README.md``. Tests marked ``serial`` must not run in
parallel (do not combine with ``-n``). The tests under ``tests/training/`` double as
worked examples.
