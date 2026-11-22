"""Imports between the packages only point down the layers.

``tracking`` and ``poco`` are the two fitting engines and never import each other;
both build on ``fitting``, which builds on ``machine``. Imports under
``TYPE_CHECKING`` count too: a layer should not need to know about the ones above.

Only ``accelmo/backend/<name>/`` imports that backend's library, and nothing outside
``accelmo/backend`` imports a backend subpackage: the rest reaches it through
``accelmo.backend.get_backend``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import accelmo

PACKAGE_ROOT = Path(accelmo.__file__).parent

#: Packages each package must never import.
FORBIDDEN = {
    "machine": {"backend", "fitting", "tracking", "poco", "analysis", "momentum_reference"},
    "optimisers": {"backend", "machine", "fitting", "tracking", "poco"},
    "fitting": {"tracking", "poco", "momentum_reference"},
    "tracking": {"poco", "momentum_reference"},
    "poco": {"tracking", "momentum_reference"},
    # Run logs and the monitor know nothing about machines or fitters: the machine
    # side plugs in through ``accelmo.monitor.OpticsSource``.
    "runlog": {"backend", "machine", "fitting", "tracking", "poco", "monitor"},
    "monitor": {"backend", "machine", "fitting", "tracking", "poco"},
    "backend": {"tracking", "poco", "momentum_reference"},
}

#: Libraries owned by one backend subpackage.
BACKEND_LIBRARIES = {"madng": ("pymadng", "pymadng_utils.mad"), "xtrack": ("xtrack", "xobjects", "xpart")}
#: Name-table imports of backend libraries allowed anywhere (constants, no model code).
SHARED_NAME_TABLES = {
    ("pymadng_utils.mad.accelerator_mad_interface", "MAGNET_STRENGTH_SUFFIXES"),
    ("pymadng_utils.mad.ac_kickers", "AC_KICKERS"),
}


def _imported_packages(path: Path) -> set[str]:
    packages = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "accelmo" and len(parts) > 1:
                packages.add(parts[1])
    return packages


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_imports_only_lower_layers(package: str) -> None:
    violations = {
        str(path.relative_to(PACKAGE_ROOT)): sorted(_imported_packages(path) & FORBIDDEN[package])
        for path in (PACKAGE_ROOT / package).rglob("*.py")
    }
    assert {path: found for path, found in violations.items() if found} == {}


def _module_imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    imports = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
    return imports


def _owner(module: str) -> str | None:
    for backend, libraries in BACKEND_LIBRARIES.items():
        if any(module == lib or module.startswith(f"{lib}.") for lib in libraries):
            return backend
    return None


def test_backend_libraries_stay_in_their_backend() -> None:
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        parts = path.relative_to(PACKAGE_ROOT).parts
        home = parts[1] if parts[0] == "backend" and len(parts) > 2 else None
        for module, names in _module_imports(path):
            owner = _owner(module)
            if owner is not None and owner != home:
                if names and all((module, name) in SHARED_NAME_TABLES for name in names):
                    continue
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
            if module.startswith("accelmo.backend.") and parts[0] != "backend":
                if module.split(".")[2] not in ("protocol",):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
    assert violations == []
