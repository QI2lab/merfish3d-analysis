"""Datastore access helpers for the viewer."""

from pathlib import Path
from typing import Any

import numpy as np

from merfish3danalysis.qi2labDataStore import qi2labDataStore


def normalize_datastore_path(path: Path) -> Path:
    """
    Resolve an experiment root or direct datastore path to a datastore path.

    Parameters
    ----------
    path : Path
        path for this viewer operation.

    Returns
    -------
    Path
        Computed viewer result.
    """

    expanded = path.expanduser().resolve()
    direct_state_path = expanded / "datastore_state.json"
    if direct_state_path.exists():
        return expanded

    nested = expanded / "qi2labdatastore"
    nested_state_path = nested / "datastore_state.json"
    if nested_state_path.exists():
        return nested

    raise FileNotFoundError(
        "Could not find qi2lab datastore. Select an experiment root containing "
        "'qi2labdatastore' or select the datastore directory directly."
    )


def open_datastore(datastore_path: Path) -> Any:
    """
    Open a qi2lab datastore without expensive validation.

    Parameters
    ----------
    datastore_path : Path
        datastore_path for this viewer operation.

    Returns
    -------
    Any
        Computed viewer result.
    """

    return qi2labDataStore(datastore_path, validate=False)


def component_summary(datastore: Any) -> dict[str, bool]:
    """
    Return datastore component availability from existing datastore state.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    dict[str, bool]
        Computed viewer result.
    """

    state = datastore.datastore_state or {}
    return {
        "Calibrations": bool(state.get("Calibrations", False)),
        "Corrected": bool(state.get("Corrected", False)),
        "LocalRegistered": bool(state.get("LocalRegistered", False)),
        "GlobalRegistered": bool(state.get("GlobalRegistered", False)),
        "Fused": bool(state.get("Fused", False)),
        "SegmentedCells": bool(state.get("SegmentedCells", False)),
        "DecodedSpots": bool(state.get("DecodedSpots", False)),
        "FilteredSpots": bool(state.get("FilteredSpots", False)),
    }


def decoded_available(datastore: Any) -> bool:
    """
    Return whether decoded spots are available without requiring fresh state flags.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    bool
        Computed viewer result.
    """

    state = component_summary(datastore)
    return state["DecodedSpots"] or state["FilteredSpots"]


def cell_outlines_available(datastore: Any) -> bool:
    """
    Return whether cell outlines are available without requiring fresh state flags.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    bool
        Computed viewer result.
    """

    return component_summary(datastore)["SegmentedCells"]


def global_fused_available(datastore: Any) -> bool:
    """
    Return whether a fused global polyDT image appears to be available.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    bool
        Computed viewer result.
    """

    return component_summary(datastore)["Fused"]


def global_cellpose_segmentation_available(datastore: Any) -> bool:
    """
    Return whether a global polyDT Cellpose segmentation image is available.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    bool
        Computed viewer result.
    """

    return component_summary(datastore)["SegmentedCells"]


def codebook_gene_bits(datastore: Any) -> dict[str, list[str]]:
    """
    Map codebook genes to existing datastore bit IDs.

    Parameters
    ----------
    datastore : Any
        datastore for this viewer operation.

    Returns
    -------
    dict[str, list[str]]
        Computed viewer result.
    """

    parsed = datastore.load_codebook_parsed()
    if parsed is None:
        return {}

    gene_ids, codebook_matrix = parsed
    bit_ids = list(datastore.bit_ids or [])
    codebook_array = np.asarray(codebook_matrix)
    gene_to_bits: dict[str, list[str]] = {}
    for gene_id, row in zip(gene_ids, codebook_array, strict=False):
        selected_bits = [
            bit_ids[bit_idx]
            for bit_idx, value in enumerate(np.asarray(row).astype(bool))
            if value and bit_idx < len(bit_ids)
        ]
        gene_to_bits[str(gene_id)] = selected_bits

    return gene_to_bits


def unavailable_data_message(error: ValueError) -> str:
    """
    Return a user-facing message for unavailable viewer data.

    Parameters
    ----------
    error : ValueError
        error for this viewer operation.

    Returns
    -------
    str
        Computed viewer result.
    """

    return f"Data not available: {error}"
