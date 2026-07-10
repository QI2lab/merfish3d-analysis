"""Datastore access helpers for the viewer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@dataclass(frozen=True)
class ViewerDatastoreOptions:
    """Datastore-derived options needed by the viewer controller."""

    components: dict[str, bool]
    tile_ids: tuple[str, ...]
    round_ids: tuple[str, ...]
    bit_ids: tuple[str, ...]
    proseg_runs: tuple[str, ...]
    baysor_available: bool
    transcript_gene_to_bits: dict[str, list[str]]


@dataclass(frozen=True)
class ViewerDatastoreLoadResult:
    """Opened datastore and controller options for one path."""

    datastore_path: Path
    datastore: Any
    options: ViewerDatastoreOptions


def normalize_datastore_path(path: Path) -> Path:
    """
    Resolve an experiment root or direct datastore path to a datastore path.

    Parameters
    ----------
    path : Path
        Experiment root or qi2lab datastore directory.

    Returns
    -------
    Path
        Resolved qi2lab datastore directory.
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
        qi2lab datastore directory.

    Returns
    -------
    Any
        Opened qi2lab datastore.
    """
    return qi2labDataStore(datastore_path, validate=False)


def component_summary(datastore: Any) -> dict[str, bool]:
    """
    Return datastore component availability from existing datastore state.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.

    Returns
    -------
    dict[str, bool]
        Component names mapped to availability flags.
    """
    state = datastore.datastore_state or {}
    return {
        "Calibrations": bool(state.get("Calibrations", False)),
        "Corrected": bool(state.get("Corrected", False)),
        "LocalRegistered": bool(state.get("LocalRegistered", False)),
        "GlobalRegistered": bool(state.get("GlobalRegistered", False)),
        "Fused": bool(state.get("Fused", False)),
        "SegmentedCells": bool(state.get("SegmentedCells", False)),
        "Transcripts": bool(state.get("DecodedSpots", False))
        or bool(state.get("FilteredSpots", False)),
    }


def codebook_gene_bits(datastore: Any) -> dict[str, list[str]]:
    """
    Map codebook genes to existing datastore bit IDs.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.

    Returns
    -------
    dict[str, list[str]]
        Gene names mapped to active bit identifiers.
    """
    parsed = datastore.load_codebook_parsed()
    if parsed is None:
        return {}

    gene_ids, codebook_matrix = parsed
    bit_ids = list(datastore.bit_ids or [])
    codebook_array = np.asarray(codebook_matrix)
    transcript_gene_to_bits: dict[str, list[str]] = {}
    for gene_id, row in zip(gene_ids, codebook_array, strict=False):
        selected_bits = [
            bit_ids[bit_idx]
            for bit_idx, value in enumerate(np.asarray(row).astype(bool))
            if value and bit_idx < len(bit_ids)
        ]
        transcript_gene_to_bits[str(gene_id)] = selected_bits

    return transcript_gene_to_bits


def viewer_datastore_options(datastore: Any) -> ViewerDatastoreOptions:
    """
    Return datastore-derived selector options for the viewer controller.

    Parameters
    ----------
    datastore : Any
        qi2lab datastore-like object.

    Returns
    -------
    ViewerDatastoreOptions
        Immutable selector metadata for the controller.
    """
    return ViewerDatastoreOptions(
        components=component_summary(datastore),
        tile_ids=tuple(str(tile) for tile in datastore.tile_ids or ()),
        round_ids=tuple(str(round_id) for round_id in datastore.round_ids or ()),
        bit_ids=tuple(str(bit_id) for bit_id in datastore.bit_ids or ()),
        proseg_runs=tuple(datastore.list_proseg_3d_runs()),
        baysor_available=bool(datastore.baysor_3d_available()),
        transcript_gene_to_bits=codebook_gene_bits(datastore),
    )


def load_datastore_for_viewer(path: Path) -> ViewerDatastoreLoadResult:
    """
    Open a datastore and collect viewer selector metadata.

    Parameters
    ----------
    path : Path
        Experiment root or qi2lab datastore directory.

    Returns
    -------
    ViewerDatastoreLoadResult
        Open datastore and controller selector metadata.
    """
    datastore_path = normalize_datastore_path(path)
    datastore = open_datastore(datastore_path)
    return ViewerDatastoreLoadResult(
        datastore_path=datastore_path,
        datastore=datastore,
        options=viewer_datastore_options(datastore),
    )
