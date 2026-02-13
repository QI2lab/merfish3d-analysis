"""
qi2lab 3D MERFISH GPU processing.

This package provides tools for processing 3D MERFISH data using GPU
acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__author__ = "Douglas Shepherd"
__email__ = "douglas.shepherd@asu.edu"

# Default/fallback version; may be overwritten below by package metadata.
__version__ = "0.8.0"

import importlib
import importlib.util as _ilus
import pkgutil as _pkgutil

if TYPE_CHECKING:
    from types import ModuleType

__all__: list[str] = []

# Discover first-level submodules/subpackages without importing them.
# `__path__` exists for packages, but may be missing in atypical contexts.
try:
    _names: list[str] = [m.name for m in _pkgutil.iter_modules(__path__)]  # type: ignore[name-defined]
except (NameError, OSError):
    _names = []

# Only advertise names that resolve to a module/package spec.
for _name in _names:
    if _ilus.find_spec(f"{__name__}.{_name}") is not None:
        __all__.append(_name)


def _lazy_import(name: str) -> ModuleType:
    """Import and cache a submodule the first time it is accessed.

    Parameters
    ----------
    name
        The immediate child module/subpackage name to import.

    Returns
    -------
    types.ModuleType
        The imported module.
    """
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    """Provide lazy access to immediate submodules.

    Parameters
    ----------
    name
        Attribute requested from this package.

    Returns
    -------
    Any
        The lazily imported module if `name` is a discoverable submodule.

    Raises
    ------
    ImportError
        If `name` is discoverable but fails to import.
    AttributeError
        If `name` is not a discoverable attribute/submodule.
    """
    if name in __all__:
        try:
            return _lazy_import(name)
        except Exception as exc:
            raise ImportError(
                f"Optional submodule '{__name__}.{name}' could not be imported; "
                "it may require environment-specific dependencies."
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return attributes for introspection tools (e.g. dir()).

    Returns
    -------
    list[str]
        Sorted attribute names including globals and lazily exposed submodules.
    """
    return sorted(list(globals().keys()) + __all__)


# Soft version resolution (doesn't fail in editable/dev installs).
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("merfish3danalysis")
    except PackageNotFoundError:
        __version__ = "0+src"
except ImportError:
    # Extremely old Python environments only; Python 3.12 should never hit this.
    __version__ = "0+unknown"
except Exception:
    # Defensive: avoid import-time hard failures due to metadata edge cases.
    __version__ = "0+unknown"
