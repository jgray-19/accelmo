Early proof of concept: AdamW in MAD-NG (2025)
================================================

This page records the first study behind accelmo, a talk by Joshua Gray and Félix Carlier
(May 2025). It describes the method as it was then, not the current code: the optimisers
now run in Python (:doc:`fit_families`), Levenberg-Marquardt is the usual choice, and
measurement uncertainties are propagated and used as weights.

Test lattice
------------

* A ring of 40 m made of 4 identical cells. Each cell has a focusing and a defocusing
  quadrupole (:math:`k_1 = \pm 0.1`), 2 dipoles, a focusing and a defocusing sextupole
  (:math:`k_2 = \pm 1`) and 6 BPMs, one after each magnet.
* Every quadrupole and sextupole gets a random error of about :math:`\pm 10\%` of its
  strength. The dipoles are left unchanged.

Loss and gradient
-----------------

The loss compares the BPM readings of the ring with errors to those of the model:

.. math::

   L = \frac{1}{2} \sum_\mathrm{BPMs}
       \left[ (x_\mathrm{err} - x_\mathrm{model})^2 + (y_\mathrm{err} - y_\mathrm{model})^2 \right],

so at each BPM :math:`\partial L / \partial x = x_\mathrm{err} - x_\mathrm{model}` (and the same in
:math:`y`). The gradient with respect to the magnet strengths :math:`k_n` follows from the chain rule,

.. math::

   \underbrace{\frac{\partial L}{\partial k_n}}_{1 \times N_\mathrm{param}}
   = \underbrace{\frac{\partial L}{\partial x_\mathrm{BPM}}}_{1 \times N_\mathrm{BPM}}
     \underbrace{\frac{\partial x_\mathrm{BPM}}{\partial k_n}}_{N_\mathrm{BPM} \times N_\mathrm{param}}
   + \frac{\partial L}{\partial y_\mathrm{BPM}} \frac{\partial y_\mathrm{BPM}}{\partial k_n}.

The derivatives :math:`\partial x_\mathrm{BPM} / \partial k_n` come from MAD-NG: the strengths are
added as parameters of the tracked coordinates (GTPSA), the particles are tracked through the thick
maps for several turns, and the derivatives are read from the coordinates at each BPM. accelmo still
computes its Jacobians this way.

Method
------

AdamW, as documented by PyTorch, was written as a single MAD-NG function called once per
iteration, taking only the gradient and the parameters. The fit ran in two stages:

1. Quadrupoles only, against one on-momentum track.
2. Quadrupoles and sextupoles together, against two tracks at :math:`\delta p = \pm 3 \times 10^{-4}`.

Each stage stopped when the loss, or its change, fell below a threshold, or after a fixed number of
epochs. The initial coordinates :math:`(x, p_x, y, p_y, t, \delta p)` were

.. list-table::
   :header-rows: 1

   * - Track
     - :math:`x`
     - :math:`p_x`
     - :math:`y`
     - :math:`p_y`
     - :math:`t`
     - :math:`\delta p`
   * - On momentum
     - :math:`1.5 \times 10^{-3}`
     - :math:`-1 \times 10^{-4}`
     - :math:`-1 \times 10^{-3}`
     - :math:`1 \times 10^{-4}`
     - 0
     - 0
   * - Off momentum +
     - :math:`1.5 \times 10^{-3}`
     - :math:`-1 \times 10^{-4}`
     - :math:`-1 \times 10^{-3}`
     - :math:`1 \times 10^{-4}`
     - 0
     - :math:`3 \times 10^{-4}`
   * - Off momentum −
     - :math:`1.5 \times 10^{-3}`
     - :math:`-1 \times 10^{-4}`
     - :math:`-1 \times 10^{-3}`
     - :math:`1 \times 10^{-4}`
     - 0
     - :math:`-3 \times 10^{-4}`

Results
-------

True strengths of the ring with errors, per cell:

.. list-table::
   :header-rows: 1

   * - Magnet
     - 1
     - 2
     - 3
     - 4
   * - Focusing quadrupole
     - 0.0972
     - 0.103
     - 0.105
     - 0.0932
   * - Defocusing quadrupole
     - −0.0987
     - −0.0936
     - −0.105
     - −0.0928
   * - Focusing sextupole
     - 1.08
     - 0.918
     - 0.994
     - 0.903
   * - Defocusing sextupole
     - −1.03
     - −1.10
     - −0.909
     - −0.941

Relative error of the fitted strengths after stage 1 (quadrupoles, on momentum; 2500 iterations,
about 40 s): all below 0.1 %.

.. list-table::
   :header-rows: 1

   * - Relative error
     - 1
     - 2
     - 3
     - 4
   * - Focusing quadrupole
     - :math:`2.25 \times 10^{-4}`
     - :math:`3.05 \times 10^{-4}`
     - :math:`2.88 \times 10^{-4}`
     - :math:`6.81 \times 10^{-4}`
   * - Defocusing quadrupole
     - :math:`7.20 \times 10^{-5}`
     - :math:`3.49 \times 10^{-4}`
     - :math:`5.00 \times 10^{-4}`
     - :math:`3.03 \times 10^{-4}`

After stage 2 (quadrupoles and sextupoles, :math:`\delta p = \pm 0.03\,\%`; 30,000 iterations, under
10 minutes): all below 0.5 %.

.. list-table::
   :header-rows: 1

   * - Relative error
     - 1
     - 2
     - 3
     - 4
   * - Focusing quadrupole
     - :math:`2.30 \times 10^{-6}`
     - :math:`3.41 \times 10^{-5}`
     - :math:`4.66 \times 10^{-7}`
     - :math:`1.02 \times 10^{-5}`
   * - Defocusing quadrupole
     - :math:`3.37 \times 10^{-5}`
     - :math:`6.02 \times 10^{-6}`
     - :math:`7.80 \times 10^{-6}`
     - :math:`2.47 \times 10^{-5}`
   * - Focusing sextupole
     - :math:`4.28 \times 10^{-3}`
     - :math:`1.36 \times 10^{-3}`
     - :math:`3.29 \times 10^{-3}`
     - :math:`1.84 \times 10^{-3}`
   * - Defocusing sextupole
     - :math:`3.15 \times 10^{-3}`
     - :math:`3.32 \times 10^{-3}`
     - :math:`2.62 \times 10^{-3}`
     - :math:`1.94 \times 10^{-3}`

Conclusions at the time
-----------------------

* The study was a working proof of concept of AdamW in MAD-NG, in simulation.
* Propagating the measurement uncertainties and using them as loss weights was still to be done.
* Errors this small need precise BPM readings, a possible drawback on real data.
