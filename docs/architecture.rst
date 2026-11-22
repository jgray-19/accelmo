Architecture
============

Process model
-------------

Each fitter runs in a main process and a set of worker processes. A worker is a
``multiprocessing.Process`` that owns one MAD-NG instance. Workers do the expensive
work (tracking or closed-orbit solves, and the derivatives with respect to the knobs);
the main process sums their replies and updates the knobs.

There are two fitting engines, built the same way:

- **Tracking** (``tracking``): ``TrackingFitter`` and its subclasses
  ``ArcByArcFitter``, ``KickerFitter``, ``ACDMarkerFitter``, ``ACDSinusoidFitter`` and
  ``ACDTurnSinusoidFitter`` fit
  turn-by-turn data with mini-batch gradient descent (Adam or L-BFGS, ``tracking.sgd``) or
  Levenberg-Marquardt (``fitting.lm_loop``), optionally with the start coordinates.
- **POCO** (``poco``): ``ClosedOrbitFitter``, ``ClosedTwissFitter`` and
  ``CalibratedClosedOrbitFitter`` fit closed-orbit and closed-twiss data with
  Levenberg-Marquardt (``fitting.lm_loop``).

Both compose a :class:`~accelmo.fitting.setup.MachineSetup`: the MAD-NG model and its BPMs,
the knobs and their starting values, the output settings and the one
``SimulationConfig`` every other part reads. A tracking fit adds its BPM ranges
(``tracking.ranges``).

Worker protocol
---------------

Messages are typed dataclasses in :mod:`accelmo.fitting.protocol`. Every worker runs
the same loop, :meth:`AbstractWorker.run <accelmo.fitting.worker.AbstractWorker.run>`:

1. Receive ``Start(knobs)`` and build the MAD-NG session.
2. Answer each ``Evaluate(knobs, batch)`` with one ``GradReply`` (loss, gradient and,
   for POCO workers, the Hessian and normal matrix), and each command (``Validate``,
   ``Diagnose``, ``SetKnobs``, ``UpdateInitCoords``, ``SetUncertaintyMode``) with one
   ``Ack`` or ``LossReply``.
3. On ``Stop``, a training tracking worker replies with its ``UncertaintyPart``, then exits.

A worker sends exactly one reply per request. A failure sends one ``ErrorReply``
instead, and the worker exits without sending anything else; the main process raises
it. A NaN loss marks a worker that lost its particles or closed orbit: it is left out
of the sums and of the averaged loss
(:func:`~accelmo.fitting.reduction.reduce_replies`).

Run logs and monitoring
-----------------------

Every fit writes a run log (:mod:`accelmo.runlog`) under
``OutputConfig.run_log_root``, one directory per fit:

``stage.json``
    Name, start time, knob names and ``OutputConfig.run_metadata`` (how to rebuild
    the model). Written once.
``history.jsonl``
    One JSON object per step: ``step``, ``time`` and the logged scalars.
``knobs.f64``
    Raw float64 rows ``[step, knob_0, knob_1, ...]``; step -1 holds the starting knobs.
``result.json``
    How the fit ended. Written when it finishes.

Optics are not stored. :mod:`accelmo.monitor` recomputes them from the knobs on demand,
caches them, and serves live plots of a running or finished fit
(``python -m accelmo.monitor RUN_LOG_ROOT --spec module:SPEC``). The machine side
supplies an :class:`~accelmo.monitor.source.OpticsSource` (how to build the model and
twiss it) and any machine-specific panels, bundled in a
:class:`~accelmo.monitor.source.MonitorSpec`.

Run lifecycle
-------------

:func:`~accelmo.fitting.lifecycle.run_with_workers` gives every fitter the same shape: run the
body; on Ctrl-C return the best result so far; always stop the workers and close the
run log. For a tracking fit the body is:

1. Load the data (:class:`~accelmo.tracking.data_manager.DataManager`), split the turns into training and held-out
   validation batches with a seeded shuffle, and clamp ``num_batches`` to the data.
2. Start a :class:`~accelmo.tracking.session.TrackingSession`: build the worker
   payloads, start the training and validation
   :class:`~accelmo.fitting.pool.WorkerPool`.
3. Run :class:`~accelmo.tracking.sgd.loop.SGDLoop`: per epoch, evaluate every batch and take an optimiser step after
   each; score the held-out validation loss; keep the knobs with the lowest loss
   (validation when available); stop on a converged loss or gradient.
4. Set the workers to the best knobs and stop them; their uncertainty parts give
   ``Cov = A⁻¹ B A⁻¹`` (:func:`~accelmo.fitting.uncertainty.sandwich_uncertainties`).

Each fit returns a :class:`~accelmo.fitting.results.FitResult`: knobs, 1-sigma
uncertainties keyed by the same knob names, and
:class:`~accelmo.fitting.results.FitDiagnostics` (why and after how many iterations it stopped).

Package layout
--------------

Imports only point down this list; the two engines never import each other
(``tests/test_package_layers.py`` checks it).

``accelmo.monitor``
    Live plots served from run logs; optics recomputed from the logged knobs. Imports
    only ``runlog``; the machine side plugs in through ``OpticsSource``.

``accelmo.momentum_reference``
    Momentum reference from blank measurements; uses both engines.
``accelmo.tracking``
    The tracking engine: fitters, ``TrackingSession``, BPM ranges, ``DataManager``,
    the tracking worker and its uncertainty parts. ``tracking.dispatch`` builds the
    worker payloads and plans turns; ``tracking.sgd`` holds the SGD
    loop, learning-rate schedule and checkpointing; ``tracking.config`` the
    measurement, kicker and checkpoint settings and the tracking plans.
``accelmo.poco``
    The POCO engine: Levenberg-Marquardt fitters for closed-orbit and closed-twiss
    data, priors, BPM-gain and corrector calibration, and their workers
    (``poco.workers``).
``accelmo.fitting``
    What both engines share: ``MachineSetup``, ``SequenceConfig`` and
    ``OutputConfig``, the worker protocol and base worker, the shared-memory
    reference, the worker pool, run lifecycle, reply reduction, loss weights,
    uncertainties and ``FitResult``.
``accelmo.optimisers``
    Adam, L-BFGS and Levenberg-Marquardt.
``accelmo.machine``
    ``machine.accelerators``: machine definitions (``LHC``, ``PSB``, ``SPS``,
    ``FCC``), knob families, BPM pattern and tune configuration. ``backend.madng``: the
    MAD-NG interfaces; ``MadInterface`` builds the model,
    ``MadInterface`` adds the optimisation knobs and derivatives, and
    ``machine_state`` handles fixed machine-state inputs.
``accelmo.analysis``
    Degeneracy checks.
``accelmo.runlog``
    Run-log writer (:class:`~accelmo.runlog.writer.RunLog`) and reader
    (:class:`~accelmo.runlog.reader.StageLog`, :func:`~accelmo.runlog.reader.find_stages`).
``accelmo.config``
    ``OptimiserConfig`` and ``SimulationConfig``. Imports no other ``accelmo`` module.

Knob conventions
----------------

Fitters work in a single *optimisation space*. Inputs (``initial_knob_strengths``,
``true_strengths``), internal algorithms and reported results all use the same knob
coordinates.

dp/p and pt are related by ``pymadng_utils.physics`` (or ``accelerator.dp2pt``).
Momentum offsets are carried through MAD-NG ``pt``.

Quadrupole tilts are fitted through ``<element>.tilt`` knobs, seeded at 1e-9 rad because
a zero-angle rotation does not carry a differentiable parameter in MAD-NG.
