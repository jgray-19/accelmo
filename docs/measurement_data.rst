Measurement Data
================

This page describes the files a tracking fit reads and the machine state a model is put in.
The fitters that read them are listed in :doc:`fit_families`.

Measurement files
-----------------

A tracking fit reads one parquet file per measurement. Each row is one marker on one turn.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Column
     - Unit
     - Content
   * - ``name``
     -
     - BPM or marker name, as in the model sequence
   * - ``turn``
     -
     - Turn number. Any integer offset; the fitter shifts each file to start after the
       previous one
   * - ``bunch_number``
     -
     - Bunch the turn belongs to. Tracks never cross from one bunch to the next
   * - ``x``, ``y``
     - m
     - Position
   * - ``px``, ``py``
     - 1
     - Momentum, MAD-NG canonical (:math:`p_x / p_0`)
   * - ``var_x``, ``var_y``
     - m²
     - Variance of the position
   * - ``var_px``, ``var_py``
     - 1
     - Variance of the momentum

How the fitter reads a file:

* Only rows whose ``name`` is a BPM in the fitted range, or a marker the fitter needs, are
  read.
* A missing row, or a missing value, is not compared: the reading gets weight zero.
* A variance that is zero, negative or not finite also gives weight zero.
* The momenta are compared only with ``SimulationConfig.optimise_momenta=True``.
* Start turns per bunch: arc-by-arc and AC-dipole marker fits skip the first and last
  ``n_run_turns`` turns (at least one); kicker fits start one track at the bunch's first turn;
  ``ACDSinusoidFitter`` starts a track every ``n_run_turns`` turns.
* The file's excited plane (``x``, ``y`` or ``xy``) is read from the data: a plane whose
  coordinates span less than 1e-12, or less than a tenth of the other plane's span, counts as
  not excited. Only the excited planes are compared.

Marker rows
~~~~~~~~~~~

Some fitters start the tracking at a marker instead of a BPM. The file must then contain rows
for that marker.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Fitter
     - Marker rows
   * - :class:`~accelmo.tracking.fitter.KickerFitter`
     - ``KickerConfig.kicker_name``, with ``x``, ``px``, ``y``, ``py``
   * - :class:`~accelmo.tracking.fitter.ACDMarkerFitter`,
       :class:`~accelmo.tracking.fitter.ACDSinusoidFitter`
     - ``<acd>_before`` and ``<acd>_after``, where ``<acd>`` is the accelerator's
       ``ac_dipole_name``

Turn boundary
~~~~~~~~~~~~~

The fitter puts the BPMs of each file in ring order, starting from the BPM each recorded turn
begins at. By default this is the first BPM row in the file. Set
``MeasurementDetails.first_bpm`` when the file's row order does not start there, for example
when marker rows are written after the BPMs.

Describing the files
~~~~~~~~~~~~~~~~~~~~

:class:`~accelmo.tracking.config.models.MeasurementConfig` maps each file to a
:class:`~accelmo.tracking.config.models.MeasurementDetails`:

``interface_options``
    Keyword arguments for the MAD-NG model of this file, usually ``machine_state``.
``machine_deltap``
    Momentum offset of the beam in this file.
``first_bpm``
    BPM each recorded turn begins at (see above).

.. code-block:: python

   from pathlib import Path

   from accelmo.tracking.config.models import MeasurementConfig, MeasurementDetails

   measurement_config = MeasurementConfig(
       {
           Path("on_momentum.parquet"): MeasurementDetails(
               interface_options={"machine_state": Path("knobs_on.txt")},
           ),
           Path("off_momentum.parquet"): MeasurementDetails(
               interface_options={"machine_state": Path("knobs_off.txt")},
               machine_deltap=3e-4,
           ),
       }
   )

:func:`~accelmo.tracking.config.helpers.create_arc_measurement_config` builds the same
mapping for a single file.

Machine state
-------------

A machine state is the set of name/value pairs a model is put in before fitting: quadrupole
families, tune knobs and corrector kicks. A name is either

* a MAD-X variable, such as ``kbrqf`` or ``kqtf.a12b1``, or
* an element attribute ``<element>.<attribute>``, such as ``br3.dhz2l4.kick``. Element names
  are matched case-insensitively. Setting an attribute replaces its deferred expression with
  the number.

A state can be given in three forms
(:func:`~accelmo.machine.machine_state.resolve_machine_state`):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Form
     - Content
   * - ``dict``
     - ``{"kbrqf": 0.0723, "br3.dhz2l4.kick": 1e-4}``
   * - Knobs file (any suffix but ``.tfs``)
     - One ``name<TAB>value`` pair per line
   * - TFS corrector table (``.tfs``)
     - Columns ``ename``, ``kind``, ``hkick``, ``hkick_old``, ``vkick``, ``vkick_old``. Every
       kicker whose kick differs from ``*_old`` becomes ``<ename>.kick`` (``hkick`` and
       ``vkick`` for a ``tkicker``)

:func:`~accelmo.machine.machine_state.merge_machine_states` combines several states; later
ones override earlier ones.

Closed-orbit measurements
-------------------------

The POCO fitters take their data as Python objects, not files: see
:class:`~accelmo.poco.closed_orbit.ClosedOrbitSeries` and :doc:`user_guide`.
