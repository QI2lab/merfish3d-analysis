from typing import List
import importlib
import importlib.util as _ilus
import pkgutil as _pkgutil

__all__: List[str] = []

# Note: __path__ here refers to this subpackage's path.
try:
    _names = [m.name for m in _pkgutil.iter_modules(__path__)]  # type: ignore[name-defined]
except Exception:
    _names = []

for _name in _names:
    # Only expose immediate child modules/packages that exist
    if _ilus.find_spec(f"{__name__}.{_name}") is not None:
        __all__.append(_name)

def _lazy_import(name: str):
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module

def __getattr__(name: str):
    if name in __all__:
        try:
            return _lazy_import(name)
        except Exception as exc:
            raise ImportError(
                f"Optional module '{__name__}.{name}' could not be imported; "
                f"it may require environment-specific dependencies. Original error: {exc}"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
