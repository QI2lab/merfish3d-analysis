"""Image processing, registration, and data I/O utilities."""

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType

__all__ = [module.name for module in iter_modules(__path__)]


def __getattr__(name: str) -> ModuleType:
    """Import and cache a utility module on first access."""
    if name not in __all__:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    try:
        module = import_module(f".{name}", __name__)
    except Exception as error:
        raise ImportError(
            f"Optional module '{__name__}.{name}' could not be imported; "
            "it may require environment-specific dependencies."
        ) from error
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """List package attributes and available utility modules."""
    return sorted(set(globals()) | set(__all__))
