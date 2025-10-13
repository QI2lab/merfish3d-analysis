from typing import TYPE_CHECKING
import importlib
import importlib.util as _ilus

# List the submodules you want to expose
_submodules = ("dataio", "imageprocessing", "registration", "rlgc")

# Only advertise submodules that actually exist (don’t import them yet)
__all__ = [name for name in _submodules
           if _ilus.find_spec(f"{__name__}.{name}") is not None]

def _load_submodule(name: str):
    """Import and cache a submodule on first attribute access."""
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module  # cache for subsequent lookups
    return module

def __getattr__(name: str):
    """Lazy attribute access for listed submodules."""
    if name in __all__:
        try:
            return _load_submodule(name)
        except Exception as exc:  # wrap with a clearer error
            raise ImportError(
                f"Failed to import submodule '{name}'. "
                f"This submodule may require optional dependencies. "
                f"Original error: {exc}"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    # Make dir(pkg) show lazily-exposed names too
    return sorted(list(globals().keys()) + __all__)

# Help type checkers see the names without importing at runtime
if TYPE_CHECKING:
    from . import dataio, imageprocessing, registration, rlgc  # noqa: F401
