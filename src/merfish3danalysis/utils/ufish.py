"""Resolve and load the U-FISH weights shared by registration and bead fitting."""

from pathlib import Path
from typing import Any

UFISH_MODEL_ALIASES = {
    "merfish": "finetune_models/v1.0.1-MERFISH_model.onnx",
    "seqfish": "finetune_models/v1.0.1-seqFISH_model.onnx",
    "simfish": "finetune_models/v1.0.1-simfish_model.onnx",
    "deepspot": "finetune_models/v1.0.1-deepspot_model.onnx",
    "exseq": "finetune_models/v1.0.1-ExSeq_model.onnx",
}

DEFAULT_UFISH_MODEL = "simfish"


def resolve_ufish_weights_path(model: str | Path | None) -> Path | str:
    """
    Resolve a U-FISH model alias or path without requiring U-FISH imports.

    Parameters
    ----------
    model : str | Path | None
        U-FISH model alias, weights filename, local path, or None to use the
        default model.

    Returns
    -------
    Path | str
        Existing local path when one is found; otherwise a U-FISH weights file
        name accepted by ``UFish.load_weights``.
    """
    if model is None:
        model = DEFAULT_UFISH_MODEL

    model_str = str(model).strip()
    if not model_str:
        model_str = DEFAULT_UFISH_MODEL

    model_path = Path(model_str).expanduser()
    if model_path.exists():
        return model_path

    weights_file = UFISH_MODEL_ALIASES.get(model_str.lower(), model_str)

    local_path = Path.home() / ".ufish" / weights_file
    if local_path.exists():
        return local_path

    return weights_file


def load_ufish_model(ufish: Any, model: str | Path | None = None) -> None:
    """
    Load configured U-FISH weights from an alias, local path, or weights file.

    Parameters
    ----------
    ufish : Any
        U-FISH instance.
    model : str | Path | None
        U-FISH model alias, weights filename, local path, or None to use the
        default model.

    Returns
    -------
    None
        The weights are loaded into ``ufish`` in place.
    """
    weights = resolve_ufish_weights_path(model)

    if isinstance(weights, Path):
        ufish.load_weights_from_path(weights)
    else:
        ufish.load_weights(weights_file=weights)
