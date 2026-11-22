Replacing MAD-NG
================

MAD-NG does two jobs in every fit: it computes the model prediction at the BPMs, and it
returns the derivative of that prediction with respect to every knob. This page lists what
the rest of the code needs from those two jobs and describes two other sources of them:
``xtrack`` (implemented, limited: see `xtrack backend requirements`_) and Cheetah
(https://github.com/desy-ml/cheetah, a possible later one). The backend is chosen with
``Accelerator(backend="madng" | "xtrack")``; the interface is :mod:`accelmo.backend.protocol`.

What the optimiser needs
------------------------

Python computes the loss, the gradient, the Gauss-Newton matrices, the weights and the
uncertainties. A worker only has to return arrays.

.. list-table::
   :header-rows: 1
   :widths: 18 44 38

   * - Fit
     - Input per evaluation
     - Output
   * - Tracking
     - Knob values, start coordinates ``(x, px, y, py)`` and ``pt`` of every particle
     - ``x, y, px, py`` at every observed point: ``(n_particles, n_points)``;
       ``d{o}_dk`` for each of them: ``(n_particles, n_knobs, n_points)``;
       with ``fit_start_coords``, ``d{o}_dc0``: ``(n_particles, 4, n_points)``;
       the number of lost particles
   * - Closed orbit and closed twiss
     - Knob values and the measurement ``pt``
     - BPM names in model order; values ``(n_observables, n_bpms)``; Jacobian
       ``(n_observables, n_bpms, n_knobs)``; ``None`` when the closed orbit is lost
   * - Validation (tracking)
     - Knob values, start coordinates, ``pt``
     - The observables only

The full Jacobian is needed, not only the gradient of the loss. Levenberg-Marquardt and the
sandwich uncertainties (``Cov = A⁻¹ B A⁻¹``) are built from it. Adam and L-BFGS need only
the gradient of the loss.

Where MAD-NG is called
----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - MAD-NG use
   * - ``backend/madng/optimising_mad_interface.py``, ``aba_mad_interface.py``
     - Load sequence and beam, create the knobs from the accelerator's families, observe
       and thin the BPMs, cycle the sequence, set machine state
   * - ``backend/madng/scripts.py``, ``backend/madng/mad_scripts/run_closed_twiss_init.mad``
     - Lua text for the tracking calls and the closed-twiss solve
   * - ``tracking/worker.py``
     - TPSA map per particle, ``track`` per batch, loss report
   * - ``poco/workers/closed_twiss.py``, ``closed_orbit.py``, ``calibrated.py``
     - Closed-orbit solve, parametric ``twiss`` columns, orbit Jacobian
   * - ``fitting/worker.py``
     - ``setup_mad_interface``, ``send_knobs``, ``create_base_damap``
   * - ``momentum_reference.py``
     - ``iface.mad.send(...)`` to build the reference
   * - ``machine/accelerators/*.py``
     - Element name patterns are MAD-NG Lua patterns; ``_KNOB_PREPARATION`` is Lua text

What stays unchanged
--------------------

* Fitters, the worker pool, the message protocol and the run log.
* Data loading, batching, weights and validation split.
* Optimisers and the Levenberg-Marquardt loop.
* Loss, gradient and uncertainty assembly (``_compute_loss_and_gradients``,
  ``reduce_replies``, ``sandwich_uncertainties``).

What a replacement implements
-----------------------------

1. **Model build.** Load the lattice with the beam and energy, apply the fixed machine
   state, create one named parameter per knob selected by ``errors`` and ``misalignments``,
   mark the BPMs as observation points, and start the lattice at a given element.
2. **Tracking evaluation.** The function of the first table row.
3. **Closed-orbit evaluation.** The function of the second table row.
4. **Accelerator families.** Element patterns in the form the new code understands.
5. **Momentum reference and monitor.** ``momentum_reference.py`` and the ``OpticsSource``
   of ``accelmo.monitor``.

Steps 2 and 3 replace ``TrackingWorker`` and the POCO workers. The new workers subclass
:class:`~accelmo.fitting.worker.AbstractWorker`, which needs ``prepare_data``, ``evaluate``
and the setup hooks.

Sources of the gradient
-----------------------

Only automatic differentiation is used. Finite differences are not a gradient source; the
tests use them as an independent check of the exact derivatives.

.. list-table::
   :header-rows: 1
   :widths: 16 28 30 26

   * -
     - MAD-NG (today)
     - xtrack (planned first replacement)
     - Cheetah (possible later)
   * - Derivative method
     - Knobs are TPSA parameters; one pass returns the value and every derivative
     - Knobs are TPSA parameters of ``xtrack.tpsa.ParticlesTpsa`` (``KnobParameters``); one
       parametric track per particle returns the value and every derivative
     - PyTorch autograd through the tracking
   * - Model runs per gradient
     - 1 per batch
     - 1 per particle
     - 1 forward and 1 backward pass for the loss gradient
   * - Full Jacobian
     - Included
     - Included
     - One backward pass per output, or one forward-mode pass per knob
   * - Hardware
     - CPU
     - CPU only
     - CPU or GPU (PyTorch)
   * - Closed orbit
     - ``cofind`` and ``twiss``
     - ``line.twiss(method="4d")`` plus the fixed-point shift (below)
     - Not confirmed here
   * - Evidence in this repository
     - All fits
     - None in ``accelmo``; ``../poco_fodo_study/fodo_study/xs_ring.py`` for a FODO ring
     - None

xtrack
~~~~~~

xtrack is already the reference of this repository: ``xtrack_tools`` builds the lines that
generate every test measurement. Its exact derivatives come from ``xtrack.tpsa`` with the
``madng_tpsa`` package, which is the MAD-NG method.

``../poco_fodo_study/fodo_study/xs_ring.py`` implements it for the closed-orbit
Jacobian of a FODO ring:

1. Build the line, one line variable per fitted quadrupole strength, and set
   ``line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True`` (TPSA tracking needs radiation off).
2. Make the knobs parameters: ``descriptor = madng_tpsa.Descriptor(6, 1, num_params=K,
   param_order=1)``, then ``KnobParameters(line, knob_names, descriptor).apply()``.
3. Find the closed orbit with ``line.twiss(method="4d", only_orbit=True)`` and start a
   ``ParticlesTpsa`` on it.
4. Track one turn with ``multi_element_monitor_at`` set to the BPMs and the monomials of
   interest recorded: ``R_s`` (the four transverse first-order terms) and ``P_s`` (one term
   per parameter).
5. Solve the fixed-point shift ``S = (I - R)^-1 P`` and return ``d(orbit_s)/dk = R_s S + P_s``.

The same module offers ``TpsaOptics`` (optical functions and their parameter derivatives
from a map of order 2 or more) and ``ActionTpsaTrack`` (a ``line.match`` action that returns
an analytic Jacobian).

Properties that bear on a worker:

* ``ParticlesTpsa`` is a single map: scalar coordinates only. A batch of ``P`` particles is
  ``P`` parametric tracks, run one after the other on the CPU.
* Radiation must be off.
* The reproducer ``../poco_fodo_study/scratch/xtrack_tpsa_monitor_overhead.py`` times the
  Python overhead of recording many monomials at many monitors.
* Not checked: which elements of the LHC, PSB and SPS lattices (edges, fringes, multipoles,
  kickers, ACD) the TPSA kernels support, and the time of one batch.

An xtrack worker:

1. Builds the line with ``xtrack_tools.env.initialise_env`` and sets the knobs as line
   variables.
2. Closed-orbit fits: returns the orbit (and, for closed twiss, the optics) at the BPM rows
   with the TPSA Jacobian.
3. Tracking fits: builds particles at the measured start coordinates, tracks the batch with
   one monitor per BPM, and returns the observables with the Jacobian.
4. Reports a lost particle through the lost-particle count and a lost closed orbit as
   ``None``.

Cheetah
~~~~~~~

Cheetah is a PyTorch package for differentiable beam dynamics. Its elements are
``torch.nn.Module`` objects, a lattice is a ``Segment``, a beam is a ``ParticleBeam`` or a
``ParameterBeam``, and it is vectorised with PyTorch broadcasting. Its documentation
describes the gradient of the beam position at the end of a lattice with respect to the
element strengths, and states that particle losses in ``Aperture`` are not differentiable.

A Cheetah worker would do the following:

1. Convert the lattice to a ``Segment`` and set the knob attributes as tensors with
   ``requires_grad=True``.
2. Track a ``ParticleBeam`` made of the measured start coordinates, with one observation
   point per BPM.
3. For Adam and L-BFGS: build the weighted loss in PyTorch and call ``backward`` once. The
   gradient arrives for all knobs.
4. For Levenberg-Marquardt and the uncertainties: build the Jacobian with
   ``torch.func.jacfwd`` or ``jacrev`` over the knob tensor, and return it in the contract
   shape.

Because PyTorch differentiates the loss directly, step 3 replaces both the observable
arrays and the derivative arrays for the gradient-descent optimisers. The worker reply
(``GradReply``) already carries only the loss and the gradient.

Items to check before choosing Cheetah. None of these is confirmed in this repository or in
the sources read for this page:

* Whether the element models cover what the fits use: thick and thin quadrupoles, dipoles
  with edges and fringes, sextupoles, kickers, and the misalignment and tilt attributes of
  each family in :doc:`fit_families`.
* The order of the tracking maps (linear or nonlinear) and the resulting difference from
  MAD-NG ``method=6``. The tests record the xsuite and MAD-NG agreement floor of the loss
  (``test_controller_acd.py``: about 5e-15 for the PSB ACD fit); a different model sets a
  different floor.
* Off-momentum tracking with ``pt`` per particle.
* A closed-orbit solve and a ring twiss, which the POCO fitters need. Without them a closed
  orbit has to be found with an outer solver, and its derivative needs the implicit
  function theorem.
* A path from the MAD-X sequences used here to a ``Segment``.

Checking a replacement
----------------------

Use the existing tests as the acceptance criteria.

1. **Jacobian.** Repeat ``tests/mad/test_madng_xsuite_misalignment.py`` with the new worker
   in place of MAD-NG: the orbit and the Jacobian columns must agree with the xsuite finite
   difference within the same 1e-3.
2. **Loss at the truth.** Run the pattern of ``tests/training/test_lhc_acd_pipeline.py``:
   the loss at the true strengths must be far below the loss at the nominal ones.
3. **Fit.** Run a controller test (``tests/training/controller/``) with Levenberg-Marquardt
   and compare ``relative_truth_errors`` with the MAD-NG values stored in the test comments.


xtrack backend requirements
---------------------------

``accelmo.backend.xtrack`` fits only knobs whose xsuite field is TPSA-capable. Measured on
xtrack 0.118.6 (``_float_or_tpsa_fields`` of each element class):

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Element class
     - TPSA fields
     - Consequence
   * - ``Quadrupole``
     - ``k1``, ``k1s``
     - ``quad k1`` knobs supported (``<Q>.dk1l`` adds ``dk1l / L`` to ``k1``)
   * - ``Bend``, ``RBend``
     - ``k0``, ``k1``, ``k2``
     - ``bend k0`` knobs supported (``<B>.dk0l`` adds ``dk0l / L`` to ``k0``)
   * - ``Sextupole``, ``Octupole``
     - ``k2``, ``k2s`` / ``k3``, ``k3s``
     - Not wired yet (no family uses the thick field)
   * - ``Multipole`` (thin correctors, thin sextupoles)
     - none
     - No ``kick``, ``knl[n]`` or ``ksl[n]`` knobs; corrector kicks are machine state only
   * - Any element: ``shift_x``, ``shift_y``, ``rot_s_rad``
     - none
     - No misalignment knobs (``dx``, ``dy``, ``ds``, ``tilt``)
   * - ``Cavity``, AC kickers
     - none
     - No ``volt`` knobs, no AC-kicker tracking

Other requirements of the backend:

* TPSA tracking needs ``line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True``: proton machines only.
* The TPSA kernels contain no aperture elements; the backend drops every ``Limit*`` element
  (MAD-NG tracks with a 100 m aperture).
* xsuite keeps no MAD-X kind; the backend reads each element's base kind from the
  sequence file declarations (``madx_element_kinds``).
* Not supported: backward sequences (LHC beam 2), ``sdir = -1``, fitted ``pt``, PSB cell
  grouping, LHC bend normalisation, tracking anchors (ACD/kicker markers), b2 errors,
  calibrated closed orbit, optics observables (closed orbit ``x, px, y, py`` only), a named
  range over more than one turn, and the momentum-reference table. Each raises
  ``NotImplementedError``.

Agreement with MAD-NG on PSB ring 3 (``tests/backend/test_xtrack_backend.py``,
``scratch_tmp/xt/compare_backends.py``):

* Knob names and observed BPMs: identical (48 quad knobs, 16 BPMs).
* Closed orbit from one corrector state: 5e-12 m vertical, 3e-13 m horizontal on ~1 mm; the
  corrector sign conventions agree.
* One-turn horizontal tracking: x 6e-10 m on 5 mm, d(x)/dk 6e-9 on 4e-2.
* Vertical plane, off momentum: with the bundled MAD-NG release (``mad_Linux`` 1.1.15)
  d(y)/d(y0) differs by 3e-4 relative at pt = 1e-3 (1e-6 on momentum), accumulated in the
  quadrupoles. The xsuite TPSA map equals plain xsuite tracking (2e-17 m) and finite
  differences (6e-10). With the MAD-NG dev build (``MAD/src/mad``, or ``MAD/src/mad_fringe``)
  the one-turn y difference drops from 3.2e-7 m to 4.3e-10 m, the same as x: the release
  binary's off-momentum quadrupole map differs, not xsuite.
* Closed orbit with quad knobs at ±2e-3 and correctors on: 1e-4 relative. The knob enters
  MAD-NG as ``dknl[2]`` and xsuite as a thick ``k1`` change, which are different maps.
