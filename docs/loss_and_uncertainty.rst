Loss, Weights and Uncertainties
===============================

This page gives the formulas a tracking fit uses. The POCO fitters use the same weights and
uncertainty estimate.

Loss
----

Each worker compares the model with the measurement at its BPMs and returns the chi-square

.. math::

   \chi^2 = \sum_{o} \sum_{\mathrm{particles}} \sum_{\mathrm{points}} w_o \, (m_o - d_o)^2 ,

where :math:`o` runs over the compared observables (``x`` and ``y``, plus ``px`` and ``py`` with
``optimise_momenta``), :math:`m_o` is the model value and :math:`d_o` the measured value. The
optimiser adds the worker chi-squares (:func:`~accelmo.fitting.reduction.reduce_replies`); a worker
that lost its particles returns NaN and is left out.

* Levenberg-Marquardt minimises :math:`\chi^2` plus the priors.
* Adam and L-BFGS minimise :math:`\chi^2 / N`, with :math:`N` the number of training points with
  a non-zero weight. The validation loss is the held-out :math:`\chi^2` divided by its own
  :math:`N`, so the two are on one scale.

Weights
-------

.. math::

   w_o = \frac{1}{\sigma_o^2}

* :math:`\sigma_o^2` is the ``var_*`` column of the measurement file
  (:doc:`measurement_data`).
* A missing reading, or a variance that is zero, negative or not finite, gets :math:`w_o = 0`.

Gradient
--------

The model values and their derivatives by every knob come from one MAD-NG track with the knobs
as TPSA parameters. The gradient is

.. math::

   \frac{\partial \chi^2}{\partial k_n} = 2
   \sum_{o} \sum_{\mathrm{particles}} \sum_{\mathrm{points}}
   w_o \, (m_o - d_o) \, \frac{\partial m_o}{\partial k_n} .

Levenberg-Marquardt also uses the Gauss-Newton Hessian
:math:`2 J^\mathsf{T} W J`, where :math:`J` holds the derivatives :math:`\partial m / \partial k`.

Uncertainties
-------------

After the fit, the workers send their Jacobians at the best knobs and the start coordinates they
used. The 1-sigma uncertainty of each knob
is

.. math::

   \sigma_k = \sqrt{\operatorname{diag}\left(A^{-1} B A^{-1}\right)} ,

with :math:`A = J^\mathsf{T} W J` and :math:`B` the measurement noise propagated through every
reading and start coordinate (:func:`~accelmo.fitting.uncertainty.sandwich_uncertainties`).
Eigenvalues of :math:`A` below a floor are raised to the
floor, so a direction the data do not constrain gets a large, finite uncertainty.
:meth:`~accelmo.tracking.fitter.TrackingFitter.check_degeneracy` lists those directions
before a fit.

With ``TrackingParameters.fit_start_coords`` the start coordinates are fit parameters. Each
particle's start is projected out of :math:`A` and of every reading's sensitivity (the Schur
complement of the joint knob and start matrix), and the measured start variances are not used.

Gaussian priors (``TrackingParameters.knob_sigmas``, ``bpm_gain_sigma`` and the zero-mean gain
groups) are added as one more measurement: :math:`P = 1/\sigma_\mathrm{prior}^2` enters both
:math:`A` and :math:`B`, so with :math:`B = A` the covariance is :math:`(A + P)^{-1}`.

``OutputConfig.include_uncertainty=False`` skips this step; the uncertainties are then zero.

Knob values
-----------

A fitter reports its knobs in *optimisation space*:

* A field error is the change of the integrated strength, for example ``MQ.12R3.B1.dk1l`` is
  the change of :math:`k_1 L` in m⁻¹.
* A misalignment is the offset or angle itself: ``.dx``, ``.dy``, ``.ds`` in m, ``.tilt`` in rad.
* The beam energy is fitted as ``pt``; a ``deltap`` given as a starting value is converted
  to ``pt``.

:meth:`Accelerator.format_result_knobs <accelmo.machine.accelerators.base.Accelerator.format_result_knobs>`
renames indexed multipole knobs (``knl[3]``) for reports.
