"""
qi2lab 3D MERFISH GPU processing.

This package provides tools for processing 3D MERFISH data using GPU
acceleration.
"""

__version__ = "0.7.6"
__author__ = "Douglas Shepherd"
__email__ = "douglas.shepherd@asu.edu"


from typing import List
import importlib
import importlib.util as _ilus
import pkgutil as _pkgutil

__all__: List[str] = []

# Discover first-level submodules/subpackages without importing them.
try:
    _names = [m.name for m in _pkgutil.iter_modules(__path__)]  # type: ignore[name-defined]
except Exception:
    _names = []

# Only advertise names that resolve to a module/package spec.
for _name in _names:
    if _ilus.find_spec(f"{__name__}.{_name}") is not None:
        __all__.append(_name)

def _lazy_import(name: str):
    """Import and cache a submodule the first time it is accessed."""
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module

def __getattr__(name: str):
    """Provide lazy access to immediate submodules."""
    if name in __all__:
        try:
            return _lazy_import(name)
        except Exception as exc:
            raise ImportError(
                f"Optional submodule '{name}' could not be imported; "
                f"it may require environment-specific dependencies. Original error: {exc}"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    """Ensure dir(merfish3danalysis) shows lazily-exposed names."""
    return sorted(list(globals().keys()) + __all__)

# Soft version (doesn't fail in editable/dev installs)
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("merfish3danalysis")
    except PackageNotFoundError:
        __version__ = "0+src"
except Exception:
    __version__ = "0+unknown"
