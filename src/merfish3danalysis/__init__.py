"""
qi2lab 3D MERFISH GPU processing.

This package provides tools for processing 3D MERFISH data using GPU
acceleration.
"""

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

__version__ = "0.14.0"
__author__ = "Douglas Shepherd"
__email__ = "douglas.shepherd@asu.edu"

__all__ = [module.name for module in iter_modules(__path__)]


def __getattr__(name: str) -> ModuleType:
    """Import and cache a submodule on first access."""
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    try:
        module = import_module(f".{name}", __name__)
    except Exception as error:
        raise ImportError(
            f"Optional submodule '{__name__}.{name}' could not be imported; "
            "it may require environment-specific dependencies."
        ) from error
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """List package attributes and available submodules."""
    return sorted(set(globals()) | set(__all__))
