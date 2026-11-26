"""Settings of the gradient-descent optimiser, the worker processes and the tracking fit."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

#: Optimisers :class:`OptimiserConfig` can name.
OPTIMISER_TYPES = ("adam", "lbfgs")

#: Hessians :class:`TrackingParameters` can use for the uncertainty.
UNCERTAINTY_HESSIANS = ("true", "gauss_newton")


@dataclass(frozen=True)
class OptimiserConfig:
    """Configuration for gradient descent optimiser algorithms.

    Controls learning rates, convergence criteria, and optimiser type
    for the gradient descent training process.
    """

    max_epochs: int
    warmup_epochs: int
    warmup_lr_start: float
    max_lr: float
    min_lr: float
    optimiser_type: str = field(default="adam")  # one of OPTIMISER_TYPES

    # Smoothing factor of the moving average of the relative loss change the stopping rule uses.
    loss_change_alpha: float = field(default=0.2)
    # Stop once the smoothed relative loss change falls below this, after at least
    # loss_change_min_epoch_fraction * max_epochs epochs. 0 disables the rule.
    loss_change_tolerance: float = field(default=1e-6)
    loss_change_min_epoch_fraction: float = field(default=0.2)

    # Fit in stages of this many epochs (``max_epochs`` is then their total): the
    # learning-rate schedule (warmup, then decay over the rest of the stage) restarts at
    # each stage and the best loss and stopping averages reset. The stopping rules are
    # off, so a stage always runs its epochs and the next starts at a multiple of this.
    stage_epochs: int | None = field(default=None)

    # L-BFGS-specific parameters (ignored for adam)
    lbfgs_history_size: int = field(default=10)
    lbfgs_max_step_norm: float | None = field(default=1.0)
    lbfgs_powell_damping: float = field(default=0.2)

    # Adam-specific parameters (ignored for lbfgs)
    adam_weight_decay: float = field(default=0.0)
    # None keeps each optimiser's own default. Lower it when the gradients are far
    # below the default, or eps rather than the gradient scale sets the step size.
    adam_eps: float | None = field(default=None)

    def __post_init__(self):
        if self.optimiser_type not in OPTIMISER_TYPES:
            raise ValueError(
                f"Unknown optimiser type {self.optimiser_type!r}; choose from {OPTIMISER_TYPES}"
            )

    @property
    def decay_epochs(self) -> int:
        """Epochs of learning-rate decay after the warmup, per stage."""
        return (self.stage_epochs or self.max_epochs) - self.warmup_epochs


@dataclass(frozen=True)
class WorkerRuntime:
    """How the worker processes run: how many, their log level and BLAS threads."""

    num_workers: int

    # Level of each worker's own log file; WARNING and above also reach the main process.
    logging_level: int = field(default=logging.DEBUG)

    # BLAS/LAPACK threads per worker process. Every worker does dense linear algebra
    # (e.g. the Gauss-Newton Hessian) while the other workers do the same, so the
    # library default (one thread per core, per worker) oversubscribes the machine:
    # 16 closed-orbit workers took 3.4x longer per Hessian and pushed the load
    # average to ~290 on a 64-core host. 1 (default) keeps one thread per worker;
    # None leaves the library default.
    blas_threads: int | None = field(default=1)

    def __post_init__(self):
        if self.num_workers < 1:
            raise ValueError(f"WorkerRuntime.num_workers must be >= 1, got {self.num_workers}")


@dataclass(frozen=True)
class TrackingDataConfig:
    """Which tracking data the fit uses and how its turns are batched."""

    # Batches each worker's turns are split into. Kicker and ACD-sinusoid fits use 1
    # whatever is set here: each worker holds a single track.
    num_batches: int

    # Fraction of the post-validation-split training data to use and
    # distribute among the workers. 1.0 (default) uses every available training
    # turn; smaller values keep that fraction of each file's turns (per-file
    # stratified sampling). Must be in (0, 1].
    data_fraction: float = field(default=1.0)

    # Fraction of the available turns held out per file as a disjoint validation
    # set used only to measure generalisation. These turns are removed from
    # training, so the validation loss is out-of-sample. `data_fraction` is applied to the remaining training
    # turns. Must be in [0, 1); 0.0 disables held-out validation.
    validation_fraction: float = field(default=0.1)

    # Whether to include momenta (px, py) in loss function
    # When False, only positions (x, y) are used for optimisation
    optimise_momenta: bool = field(default=True)

    def __post_init__(self):
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError(
                f"TrackingDataConfig.data_fraction must be in (0, 1], got {self.data_fraction}"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "TrackingDataConfig.validation_fraction must be in [0, 1), "
                f"got {self.validation_fraction}"
            )


@dataclass(frozen=True)
class TrackingParameters:
    """What a tracking fit fits besides the magnet knobs, and every prior width."""

    # Fit one gain per BPM and plane on the positions, reading = (1 + g) * position, with a
    # Gaussian prior of this width. None fits no gains.
    bpm_gain_sigma: float | None = field(default=None)

    # LM only: zero-centred Gaussian prior widths, as POCO's knob_sigmas: keyed by knob
    # name or terminal attribute (e.g. dk1l); a BPM gain named here overrides bpm_gain_sigma.
    knob_sigmas: dict[str, float] | None = field(default=None)

    # LM only: also fit every particle's start x, px, y, py.
    fit_start_coords: bool = field(default=False)

    # LM only: the starts are functions of the BPM gains, s(g) = s_ref + S (g - g_ref),
    # given by the initial-conditions callback as a StartGainMap; the gains' gradient and
    # Gauss-Newton blocks include the path through the starts (docs/start_gain_chain_rule.md).
    starts_follow_gains: bool = field(default=False)

    # LM only: planes ("x", "y") whose mean BPM gain is held at 0, for a common scale the data leave free.
    bpm_gain_zero_mean_planes: tuple[str, ...] = field(default=())

    # Uncertainty only: "true" is the full chi-square Hessian, Σ w (J Jᵀ + r ∂²m), from order-2 TPSA;
    # "gauss_newton" drops the residual term r ∂²m. The LM fit itself always uses Gauss-Newton.
    uncertainty_hessian: str = field(default="true")

    @property
    def start_derivatives(self) -> bool:
        """Whether the workers return the start-coordinate derivative blocks."""
        return self.fit_start_coords or self.starts_follow_gains

    def __post_init__(self):
        if self.uncertainty_hessian not in UNCERTAINTY_HESSIANS:
            raise ValueError(
                f"uncertainty_hessian must be one of {UNCERTAINTY_HESSIANS}, "
                f"got {self.uncertainty_hessian!r}"
            )
        if unknown := set(self.bpm_gain_zero_mean_planes) - {"x", "y"}:
            raise ValueError(f"bpm_gain_zero_mean_planes takes 'x' and 'y', got {sorted(unknown)}")
        if self.bpm_gain_zero_mean_planes and self.bpm_gain_sigma is None:
            raise ValueError("bpm_gain_zero_mean_planes needs fitted gains (bpm_gain_sigma)")


# Global schema constant for data files (this is appropriate as a global)
FILE_COLUMNS: tuple[str, ...] = (
    "name",
    "turn",
    "bunch_number",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
)
