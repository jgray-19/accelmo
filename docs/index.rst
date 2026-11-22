accelmo
=============

``accelmo`` estimates accelerator magnet errors (strengths, misalignments and
tilts) from beam measurements by gradient-based optimisation of MAD-NG models.
Supported machines are the LHC, PSB, SPS and FCC.

Two families of fit are provided:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Fit
     - Data
     - Entry points
   * - Tracking
     - Turn-by-turn BPM data tracked through the model
     - :class:`~accelmo.tracking.fitter.ArcByArcFitter`,
       :class:`~accelmo.tracking.fitter.ACDMarkerFitter`,
       :class:`~accelmo.tracking.fitter.KickerFitter`
   * - Closed orbit
     - Closed orbit and phase advance, at one or more machine states
     - :class:`~accelmo.poco.closed_orbit.ClosedOrbitFitter`,
       :class:`~accelmo.poco.calibrated.CalibratedClosedOrbitFitter`
   * - Closed twiss
     - Closed orbit, phase advance, beta and dispersion
     - :class:`~accelmo.poco.fitter.ClosedTwissFitter`

All fitters return a :class:`~accelmo.fitting.results.FitResult`.

Installation
------------

Python 3.11 or later is required.

.. code-block:: bash

   git clone https://github.com/jgray-19/accelmo.git
   cd accelmo
   pip install -e ".[test,docs,tracking]"

Optional dependency groups:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Contents
   * - ``tracking``
     - omc3, cpymad, pyarrow, psutil (required by the tracking workflows)
   * - ``monitor``
     - Panel, for the live fit monitor (``python -m accelmo.monitor``)
   * - ``test``
     - pytest, pytest-cov, pytest-xdist
   * - ``docs``
     - Sphinx and theme
   * - ``dev``
     - ruff, pre-commit

Build this documentation with:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

Companion packages
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - `pymadng-utils <https://jgray-19.github.io/pymadng-utils/>`_
     - Shared accelerator abstractions, dp/p and pt conversion
       (``pymadng_utils.physics``), knob-file I/O.
   * - `xtrack_tools <https://jgray-19.github.io/xtrack_tools/>`_
     - Tracking helpers and dataframe conversion, used by the tests.

Install them with:

.. code-block:: bash

   pip install pymadng-utils
   pip install git+https://github.com/jgray-19/xtrack_tools.git

The measurement and campaign workflows built on this package are maintained in
separate repositories (``lhc_measurements``, ``psb_md``, ``psb_loco``, ``lhc_loco``
and ``fcc_loco``) and are not documented here.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide
   fit_families
   tracking_tour
   architecture
   replacing_madng
   start_gain_chain_rule
   api_reference
