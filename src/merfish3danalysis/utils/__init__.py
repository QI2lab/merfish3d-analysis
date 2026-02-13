"""
Utilities subpackage with lazy imports.

This module dynamically discovers immediate child modules/packages and exposes
them via ``__all__``. Accessing an attribute matching a discovered child name
triggers a deferred import of that child module.

Notes
-----
- This pattern avoids importing optional or heavy dependencies at package import
  time.
- Discovered children are limited to immediate submodules/subpackages present on
  this package's ``__path__``.
"""

from __future__ import annotations

import importlib
import importlib.util as _ilus
import pkgutil as _pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

__all__: list[str] = []

# Note: __path__ refers to this subpackage's path when this file is imported as a package.
# It may not exist in some atypical import contexts (e.g., direct execution), so we guard it.
try:
    _names: list[str] = [m.name for m in _pkgutil.iter_modules(__path__)]  # type: ignore[name-defined]
except (NameError, OSError):
    _names = []

for _name in _names:
    # Only expose immediate child modules/packages that exist.
    if _ilus.find_spec(f"{__name__}.{_name}") is not None:
        __all__.append(_name)


def _lazy_import(name: str) -> ModuleType:
    """Import and cache a child module on first access.

    Parameters
    ----------
    name
        The name of an immediate child module/subpackage to import. This should
        be one of the entries discovered into ``__all__``.

    Returns
    -------
    types.ModuleType
        The imported module object.

    Notes
    -----
    The imported module is cached into ``globals()`` under ``name`` so subsequent
    attribute lookups return the cached module without re-importing.
    """
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes for lazy-loaded child modules.

    Parameters
    ----------
    name
        Attribute name requested from the package.

    Returns
    -------
    Any
        If ``name`` is in ``__all__``, returns the lazily imported module object.

    Raises
    ------
    ImportError
        If the module is discoverable (in ``__all__``) but fails to import, often
        due to missing optional dependencies.
    AttributeError
        If ``name`` is not a discoverable child module/subpackage.
    """
    if name in __all__:
        try:
            return _lazy_import(name)
        except Exception as exc:
            raise ImportError(
                f"Optional module '{__name__}.{name}' could not be imported; "
                "it may require environment-specific dependencies."
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return the list of available attributes for introspection tools.

    Returns
    -------
    list[str]
        Sorted attribute names including already-imported globals and lazily
        discoverable submodules listed in ``__all__``.
    """
    return sorted(list(globals().keys()) + __all__)
