Fit Families
============

A fit changes **knobs** until a model reproduces a measurement. This page lists what can be
measured, which fitter reads it, what is fitted and which solver runs.

What every fit does
-------------------

1. Load the machine (sequence, beam, energy) and create one knob per fitted magnet
   attribute.
2. Split the measurement across worker processes. Each worker owns one MAD-NG instance.
3. Send the current knobs to every worker. Each worker returns the model prediction at the
   BPMs and its derivative with respect to every knob.
4. Sum the weighted squared residuals into a loss and a gradient.
5. Update the knobs with the optimiser. Repeat from step 3.
6. Return the best knobs, their 1-sigma uncertainties and why the fit stopped
   (:class:`~accelmo.fitting.results.FitResult`).

Two engines
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Tracking (``accelmo.tracking``)
     - POCO (``accelmo.poco``)
   * - Measurement
     - Turn-by-turn BPM data: ``x``, ``px``, ``y``, ``py`` per turn
     - Closed orbit, or closed twiss (orbit, beta, phase, dispersion), at one or more momenta
   * - Model side
     - MAD-NG ``track`` from measured start coordinates, with the knobs as TPSA parameters
     - MAD-NG closed-orbit solve and parametric ``twiss`` with the knobs as TPSA parameters
   * - Solver
     - Adam, L-BFGS or Levenberg-Marquardt (``OptimiserConfig.optimiser_type``)
     - Levenberg-Marquardt
   * - Split of the data
     - Batches of turns, one particle per turn; held-out validation turns
     - One worker per series of measurements
   * - Result
     - ``FitResult``
     - ``FitResult``

The engines share the worker protocol, the worker pool, the weights, the uncertainty
estimate and the run log (``accelmo.fitting``). They do not import each other.

Tracking fitters
----------------

All tracking fitters are subclasses of :class:`~accelmo.tracking.fitter.TrackingFitter`.
They differ in where the particles start, which direction they are tracked, and which BPM
readings are compared.

.. list-table::
   :header-rows: 1
   :widths: 21 22 29 28

   * - Fitter
     - Start coordinates
     - Tracking
     - Readings compared
   * - :class:`~accelmo.tracking.fitter.ArcByArcFitter`
     - Measured, at the first BPM of each range
     - One turn, forward and backward over every ``bpm_start_points`` x ``bpm_end_points``
       range
     - BPMs inside each range
   * - :class:`~accelmo.tracking.fitter.ArcByArcFitter` with ``acd_excited=True``
     - Same
     - Same. A range that contains the AC dipole is tracked the other way round the ring
     - BPMs inside each range
   * - :class:`~accelmo.tracking.fitter.ACDMarkerFitter`
     - Measured, at the AC-dipole ``<acd>_before`` and ``<acd>_after`` marker rows
     - One turn, forward from ``after`` and backward from ``before``
     - All BPMs
   * - :class:`~accelmo.tracking.fitter.KickerFitter`
     - Measured, at the kicker marker rows
     - ``turns_after_kicker`` turns forward, one worker per file
     - All BPMs
   * - :class:`~accelmo.tracking.fitter.ACDSinusoidFitter`
     - Measured, at the first flattop turn of each bunch
     - ``n_run_turns`` turns forward through the model AC kickers
     - All BPMs, every turn
   * - :class:`~accelmo.tracking.fitter.ACDTurnSinusoidFitter`
     - Fitted: one state per turn at the ``before`` marker
     - One turn backward from ``before`` and forward from ``after`` (``before`` plus the
       kick)
     - All BPMs

Each fitter builds a tracking plan (``tracking/config/tracking.py``) that fixes the ranges
and the markers. The step-by-step run of one fit is in :doc:`tracking_tour`.

Measurement files
~~~~~~~~~~~~~~~~~

One parquet file per measurement, with the columns
``name, turn, bunch_number, x, px, y, py, var_x, var_y, var_px, var_py``.
:class:`~accelmo.tracking.config.models.MeasurementConfig` maps each file to a
:class:`~accelmo.tracking.config.models.MeasurementDetails` (MAD interface options,
momentum offset ``machine_deltap``, ``first_bpm``).

* ``ACDMarkerFitter`` and both sinusoid fitters need ``<acd>_before`` and ``<acd>_after``
  rows.
* ``KickerFitter`` needs ``x``, ``px``, ``y``, ``py`` at the kicker marker.

Tracking fit options
~~~~~~~~~~~~~~~~~~~~

These :class:`~accelmo.config.SimulationConfig` fields change what a tracking fit
parameterises. They apply to Levenberg-Marquardt tracking fits
(``optimiser_type="levenberg_marquardt"``).

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Effect
   * - ``fit_start_coords``
     - Every particle's start ``x, px, y, py`` becomes a parameter
   * - ``bpm_gain_sigma``
     - One gain per BPM and plane, reading = (1 + g) x position, with a Gaussian prior of
       this width
   * - ``starts_follow_gains``
     - Starts are functions of the BPM gains; the gradient includes that path
       (``docs/start_gain_chain_rule.md``)
   * - ``bpm_gain_zero_mean_planes``
     - Holds the mean gain of the listed planes at zero
   * - ``knob_sigmas``
     - Zero-centred Gaussian prior on named knobs
   * - ``optimise_momenta``
     - Includes ``px`` and ``py`` in the loss
   * - ``validation_fraction``, ``data_fraction``
     - Held-out and used fractions of the turns of each file

POCO fitters
------------

All POCO fitters are subclasses of :class:`~accelmo.poco.fitter.LMFitter` and use
Levenberg-Marquardt.

.. list-table::
   :header-rows: 1
   :widths: 27 33 40

   * - Fitter
     - Measurement
     - Fitted in addition to the knobs
   * - :class:`~accelmo.poco.closed_orbit.ClosedOrbitFitter`
     - :class:`~accelmo.poco.closed_orbit.ClosedOrbitSeries`: closed orbits (``x``, ``y``)
       and optionally the phase advances ``mu1``, ``mu2``, each with a ``pt`` and a
       ``machine_state``
     - Nothing; ``accelerator.optimise_energy`` fits ``pt``
   * - :class:`~accelmo.poco.calibrated.CalibratedClosedOrbitFitter`
     - Same
     - BPM gains and corrector kick gains, with Gaussian priors ``sigma_bpm`` and
       ``sigma_corrector``
   * - :class:`~accelmo.poco.fitter.ClosedTwissFitter`
     - Closed orbit, beta, phase and dispersion at each ``pt``
     - Nothing

``machine_state`` is a set of fixed MAD globals (quadrupole strengths, corrector kicks, tune
knobs). Planes listed in ``absolute_planes`` are fitted as absolute orbits; the other planes
are fitted as the change from the fitter's own ``machine_state``, using each measurement's
``reference`` orbit.

Knob families
-------------

``Accelerator(errors=..., misalignments=...)`` chooses what is fitted. Both are dictionaries
from a family name to a set of attributes. Each family is a group of magnets selected by a
name pattern. Elements whose ``nonzero`` attribute is zero get no knob.

.. list-table::
   :header-rows: 1
   :widths: 12 18 30 20 20

   * - Machine
     - Family
     - Element pattern
     - ``errors``
     - ``misalignments``
   * - LHC
     - ``bend``
     - ``MB%.``, ``MB[RXWAL]%w*%.``
     - ``k0``
     -
   * -
     - ``quad``
     - ``MQ%.``
     - ``k1``
     - ``dy``
   * -
     - ``other_quad``
     - ``MQ[^.TSY]``
     - ``k1``
     - ``dx``, ``dy``
   * -
     - ``sextupole``
     - ``MSS?%.``
     - ``k2``
     -
   * -
     - ``corrector``
     - ``MCB``
     - ``kick``
     -
   * - PSB
     - ``quad``
     - ``^BR%.Q[FD][OE]%d+$``
     - ``k0s``, ``k1``, ``k1s``
     - ``ds``, ``dx``, ``dy``, ``tilt``
   * -
     - ``bend``
     - ``^BR%.BHZ%d+$``, ``^BR%.BSW%d+L%d+%.%d+$``
     - ``k0``
     - ``ds``, ``dx``, ``dy``, ``tilt``
   * -
     - ``sextupole``
     - ``^BR%d+%.XNO%d+L1$``
     - ``knl[3]``
     -
   * -
     - ``skew_sextupole``
     - ``^BR%d+%.XSK[26]L4$``
     - ``ksl[3]``
     -
   * -
     - ``corrector``
     - ``^B[RE]%d+%.DHZ%d+L%d+$``, ``^B[RE]%d+%.DVT%d+L%d+$``
     - ``kick``
     -
   * -
     - ``acd``
     - ``^hackicker_[ab]$``, ``^vackicker_[ab]$``
     - ``volt``
     -
   * - SPS
     - ``quad``
     - ``^Q[FD]A?%.``
     - ``k1``
     -
   * -
     - ``sextupole``
     - ``^LS[FD]A?%.``
     - ``k2``
     -
   * - FCC
     - ``quad``
     - ``^Q[DF][123]AM?%.%d+$``
     - ``k1``
     -
   * -
     - ``corrector``
     - ``^[HV]COR_``
     - ``hkick``, ``vkick``
     -

Other fitted parameters:

``optimise_energy=True``
    Fits the beam momentum offset ``pt``.
``custom_knobs_to_optimise``
    Extra global MAD variables, by name.

Knob names follow the element name, for example ``MQ.12R3.B1.dk1l`` (integrated gradient
error) and ``<element>.tilt``. A tilt knob starts at 1e-9 rad. Values the fitter reports are
in optimisation space.

Solvers
-------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``optimiser_type``
     - Behaviour
   * - ``adam``
     - Mini-batch gradient descent: one step per batch, learning rate warmup then decay
   * - ``lbfgs``
     - Quasi-Newton step with a normalised gradient, history ``lbfgs_history_size``, step
       capped by ``lbfgs_max_step_norm``
   * - ``levenberg_marquardt``
     - Gauss-Newton step with damping, over every batch. Required by the options above that
       add parameters

``stage_epochs`` runs the fit in stages that restart the learning-rate schedule. The POCO
fitters always use Levenberg-Marquardt, configured by
:class:`~accelmo.optimisers.levenberg_marquardt.LevenbergMarquardtConfig`.
