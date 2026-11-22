"""Base accelerator class defining the interface for all accelerators."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from pymadng_utils.accelerators.base import Accelerator as BaseAccelerator

LOGGER = logging.getLogger(__name__)
_INDEXED_RESULT_MULTIPOLE_RE = re.compile(r"\.(knl|ksl)\[(\d+)\]$")

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class KnobSpec(NamedTuple):
    """Specification for a single optimisable knob type."""

    kind: str
    attribute: str
    pattern: str
    nonzero_attr: str | None
    label: str


MISALIGNMENT_ATTRS = frozenset({"dx", "dy", "ds", "tilt"})


@dataclass(frozen=True)
class MagnetFamily:
    """A group of magnets sharing name patterns and the attributes that may be fitted.

    Attributes:
        patterns: Element kind -> Lua name pattern (one entry per kind; see
            :func:`accelmo.machine.patterns.lua_to_regex`).
        errors: Field attributes that may be fitted (``k1``, ``k0s``, ``knl[3]``, ``kick`` ...).
        misalignments: Alignment attributes that may be fitted (subset of ``MISALIGNMENT_ATTRS``).
        nonzero_attr: Only create knobs on elements where this attribute is nonzero.
        attr_patterns: Per-attribute pattern overriding ``patterns`` for every kind.
    """

    patterns: Mapping[str, str]
    errors: frozenset[str] = frozenset()
    misalignments: frozenset[str] = frozenset()
    nonzero_attr: str | None = None
    attr_patterns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.misalignments - MISALIGNMENT_ATTRS or self.errors & MISALIGNMENT_ATTRS:
            raise ValueError(f"Misalignment attributes must be in {sorted(MISALIGNMENT_ATTRS)}")


def _normalise_selection(
    selection: Mapping[str, Iterable[str]] | None,
    families: Mapping[str, MagnetFamily],
    allowed: str,
) -> dict[str, frozenset[str]]:
    """Validate ``selection`` against each family's ``allowed`` attrs; drop empty entries."""
    result: dict[str, frozenset[str]] = {}
    for name, attrs in (selection or {}).items():
        if isinstance(attrs, str):
            raise TypeError(f"{allowed}[{name!r}] must be a collection of attributes, not a string")
        if name not in families:
            raise ValueError(f"Unknown magnet family {name!r}; valid: {sorted(families)}")
        attrs = frozenset(attrs)
        valid = getattr(families[name], allowed)
        if attrs - valid:
            raise ValueError(
                f"{allowed}[{name!r}] got {sorted(attrs - valid)}; valid: {sorted(valid)}"
            )
        if attrs:
            result[name] = attrs
    return result


class Accelerator(BaseAccelerator, ABC):
    """Abstract base class for accelerator definitions.

    This class encapsulates all backend-neutral machine configuration; the
    backend-specific code lives in ``accelmo.backend.<name>``.

    Subclasses declare ``FAMILIES``; callers pick what to fit with ``errors``
    and ``misalignments``, both keyed by family name.
    """

    FAMILIES: ClassVar[dict[str, MagnetFamily]] = {}

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float,
        bpm_pattern: str,
        particle: str = "proton",
        errors: Mapping[str, Iterable[str]] | None = None,
        misalignments: Mapping[str, Iterable[str]] | None = None,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
        backend: str = "madng",
        **kwargs,
    ):
        """Initialise base accelerator.

        Args:
            sequence_file: Path to the sequence file
            kinetic_energy: Particle kinetic energy in GeV
            bpm_pattern: Pattern for identifying BPMs in the sequence
            errors: Field errors to fit, keyed by ``FAMILIES`` name,
                e.g. ``{"quad": {"k1"}, "bend": {"k0"}}``
            misalignments: Alignment errors to fit, keyed by ``FAMILIES`` name,
                e.g. ``{"quad": {"dx", "dy", "ds", "tilt"}}``
            optimise_energy: Fit the beam energy (``pt``)
            custom_knobs_to_optimise: Extra global knob names to fit
            backend: Model backend of the fits (``accelmo.backend.BACKENDS``)
        """
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            particle=particle,
            **kwargs,
        )
        self.errors = _normalise_selection(errors, self.FAMILIES, "errors")
        self.misalignments = _normalise_selection(misalignments, self.FAMILIES, "misalignments")
        self.optimise_energy = optimise_energy
        self.custom_knobs_to_optimise = custom_knobs_to_optimise
        self.backend = backend

    def selection_kwargs(self) -> dict:
        """Return the constructor kwargs describing what is optimised (for ``copy_with``)."""
        return {
            "errors": self.errors,
            "misalignments": self.misalignments,
            "optimise_energy": self.optimise_energy,
            "custom_knobs_to_optimise": self.custom_knobs_to_optimise,
            "backend": self.backend,
        }

    def optimises(self, family: str, attribute: str) -> bool:
        """Return whether ``attribute`` of ``family`` is fitted (as error or misalignment)."""
        return attribute in self.errors.get(family, ()) or attribute in self.misalignments.get(
            family, ()
        )

    def get_knob_specs(self) -> list[KnobSpec]:
        """Expand the selected errors and misalignments into one spec per element kind."""
        specs = []
        for selection in (self.errors, self.misalignments):
            for name, attrs in selection.items():
                family = self.FAMILIES[name]
                for attr in sorted(attrs):
                    for kind, pattern in family.patterns.items():
                        pattern = family.attr_patterns.get(attr, pattern)
                        specs.append(
                            KnobSpec(kind, attr, pattern, family.nonzero_attr, f"{name} {attr}")
                        )
        return specs

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled."""
        return (
            bool(self.errors or self.misalignments)
            or self.optimise_energy
            or bool(self.custom_knobs_to_optimise)
        )

    @property
    def ac_dipole_name(self) -> str:
        """Return the AC-dipole exciter name for machines that define one."""
        raise NotImplementedError(f"{type(self).__name__} does not define an AC-dipole exciter")

    @property
    @abstractmethod
    def tune_variables(self) -> tuple[str, str]:
        """Return the names of the horizontal and vertical tune variables."""

    @property
    @abstractmethod
    def tune_integers(self) -> tuple[int, int]:
        """Return the integer tune values."""

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this accelerator."""
        # Use an ordered-dict trick to deduplicate labels while preserving insertion order.
        seen: dict[str, None] = {}
        for spec in self.get_knob_specs():
            seen[spec.label] = None
        if self.optimise_energy:
            seen["beam energy"] = None
        if self.custom_knobs_to_optimise:
            seen[f"custom knobs: {self.custom_knobs_to_optimise}"] = None
        if seen:
            LOGGER.info("Optimisation targets: %s", ", ".join(seen))
        else:
            LOGGER.info("No optimisation targets set.")

    @abstractmethod
    def copy_with(self, **overrides) -> Accelerator:
        """Return a new instance of the same type with selected parameters overridden."""

    def get_bend_lengths(self) -> dict[str, float] | None:
        """Return bend lengths required for accelerator-specific normalisation."""
        return None

    def normalise_true_strengths(
        self,
        true_strengths: dict[str, float],
        bend_lengths: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply accelerator-specific normalisation to true strengths.

        Args:
            true_strengths: Dictionary of true magnet strengths
            bend_lengths: Bend lengths for normalisation (optional). If None, uses
                accelerator-owned ``self.bend_lengths``.

        Returns:
            Normalised strengths dictionary (default: unchanged)
        """
        _ = bend_lengths
        return true_strengths

    def normalise_initial_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Map user-facing initial values into optimisation-space knob names."""
        return knobs

    def format_result_knob_names(self, knob_names: list[str]) -> list[str]:
        """Format knob names for result reporting.

        Args:
            knob_names: Knob names as used in optimisation

        Returns:
            Knob names adjusted for reporting (default: unchanged)
        """
        formatted = []
        for knob_name in knob_names:
            match = _INDEXED_RESULT_MULTIPOLE_RE.search(knob_name)
            if match is not None:
                table, index_str = match.groups()
                order = int(index_str) - 1
                suffix = f".dk{order}l" if table == "knl" else f".dk{order}sl"
                knob_name = _INDEXED_RESULT_MULTIPOLE_RE.sub(suffix, knob_name)
            formatted.append(knob_name)
        return formatted

    def format_result_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Map optimisation-space knob values to user-facing result names."""
        names = self.format_result_knob_names(list(knobs))
        return dict(zip(names, knobs.values(), strict=True))

    @property
    def has_ac_kickers(self) -> bool:
        """True when the model has turn-dependent AC kickers (their phase depends on the start turn)."""
        return False

    def ac_kicker_freqs(self, file_index: int) -> tuple[float, float] | None:
        """AC-kicker (h, v) drive tunes of measurement file ``file_index``."""
        return None

    def ac_kicker_volt_knobs(self, file_index: int) -> tuple[str, ...] | None:
        """Per-file volt (a, b) knob names of the AC kickers (``None``: shared knobs)."""
        return None

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return per-family override metadata keyed by family code d/q/s."""
        return {}

    @staticmethod
    @abstractmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from BPM name."""
