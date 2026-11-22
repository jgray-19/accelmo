API Reference
=============

The reference covers every module of the ``accelmo`` package. Measurement and campaign
workflows are maintained in separate repositories.


Primary Entry Points
--------------------

.. autosummary::
   :nosignatures:

   accelmo.machine.accelerators.base.Accelerator
   accelmo.machine.accelerators.lhc.LHC
   accelmo.machine.accelerators.psb.PSB
   accelmo.machine.accelerators.sps.SPS
   accelmo.machine.accelerators.fcc.FCC
   accelmo.config.OptimiserConfig
   accelmo.config.SimulationConfig
   accelmo.tracking.fitter.ArcByArcFitter
   accelmo.tracking.fitter.ACDMarkerFitter
   accelmo.tracking.fitter.KickerFitter
   accelmo.tracking.fitter.ACDSinusoidFitter
   accelmo.tracking.fitter.FitterOptions
   accelmo.tracking.config.models.MeasurementConfig
   accelmo.fitting.config.SequenceConfig
   accelmo.fitting.config.OutputConfig
   accelmo.fitting.results.FitResult
   accelmo.poco.closed_orbit.ClosedOrbitFitter
   accelmo.poco.closed_orbit.ClosedOrbitSeries
   accelmo.poco.calibrated.CalibratedClosedOrbitFitter
   accelmo.poco.fitter.ClosedTwissFitter
   accelmo.optimisers.levenberg_marquardt.LevenbergMarquardtConfig


Accelerators And Configuration
------------------------------

.. autosummary::
   :toctree: _autosummary

   accelmo.machine
   accelmo.machine.accelerators
   accelmo.machine.accelerators.base
   accelmo.machine.accelerators.lhc
   accelmo.machine.accelerators.psb
   accelmo.machine.accelerators.sps
   accelmo.machine.accelerators.fcc
   accelmo.machine.accelerators.magnet_grouping
   accelmo.machine.accelerators.selection
   accelmo.machine.patterns
   accelmo.machine.machine_state
   accelmo.machine.knob_names
   accelmo.config


Backends
--------

The engine protocols are specified in :doc:`backends`.

.. autosummary::
   :toctree: _autosummary

   accelmo.backend
   accelmo.backend.protocol
   accelmo.backend.madng
   accelmo.backend.madng.interface
   accelmo.backend.madng.engines
   accelmo.backend.madng.accelerator_hooks
   accelmo.backend.madng.machine_state
   accelmo.backend.madng.scripts
   accelmo.backend.xtrack
   accelmo.backend.xtrack.engines
   accelmo.backend.xtrack.model


Closed-Orbit Fitting (POCO)
---------------------------

.. autosummary::
   :toctree: _autosummary

   accelmo.poco
   accelmo.poco.closed_orbit
   accelmo.poco.calibrated
   accelmo.poco.fitter
   accelmo.poco.prior
   accelmo.poco.calibration
   accelmo.poco.workers
   accelmo.poco.workers.closed_orbit
   accelmo.poco.workers.closed_twiss
   accelmo.poco.workers.calibrated
   accelmo.momentum_reference


Tracking Fits
-------------

.. autosummary::
   :toctree: _autosummary

   accelmo.tracking
   accelmo.tracking.fitter
   accelmo.tracking.session
   accelmo.tracking.ranges
   accelmo.tracking.data_manager
   accelmo.tracking.worker
   accelmo.tracking.dispatch
   accelmo.tracking.dispatch.payloads
   accelmo.tracking.dispatch.setup
   accelmo.tracking.dispatch.turn_planner
   accelmo.tracking.uncertainty
   accelmo.tracking.ic_iteration
   accelmo.tracking.sgd
   accelmo.tracking.sgd.loop
   accelmo.tracking.sgd.scheduler
   accelmo.tracking.sgd.checkpointing
   accelmo.tracking.config
   accelmo.tracking.config.helpers
   accelmo.tracking.config.models
   accelmo.tracking.config.tracking


Shared Fitting Base
-------------------

.. autosummary::
   :toctree: _autosummary

   accelmo.fitting
   accelmo.fitting.setup
   accelmo.fitting.config
   accelmo.fitting.protocol
   accelmo.fitting.lm_loop
   accelmo.fitting.worker
   accelmo.fitting.shared_reference
   accelmo.fitting.pool
   accelmo.fitting.lifecycle
   accelmo.fitting.reduction
   accelmo.fitting.weights
   accelmo.fitting.bpm_gains
   accelmo.fitting.uncertainty
   accelmo.fitting.results


Optimisers And Numerical Helpers
--------------------------------

.. autosummary::
   :toctree: _autosummary

   accelmo.optimisers
   accelmo.optimisers.base
   accelmo.optimisers.adam
   accelmo.optimisers.lbfgs
   accelmo.optimisers.levenberg_marquardt


Analysis
--------

.. autosummary::
   :toctree: _autosummary

   accelmo.analysis
   accelmo.analysis.degeneracy_checker


Run Logs And Monitor
--------------------

.. autosummary::
   :toctree: _autosummary

   accelmo.runlog
   accelmo.runlog.reader
   accelmo.runlog.writer
   accelmo.monitor
   accelmo.monitor.source
   accelmo.monitor.view
   accelmo.monitor.cache
   accelmo.monitor.panels
   accelmo.monitor.app
