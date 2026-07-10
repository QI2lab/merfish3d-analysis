"""Datastore-backed cache repository for viewer display builders."""

from typing import Any

import numpy as np

from merfish3danalysis.viewer.overlays import (
    global_datastore_transcript_index,
    global_transcript_index,
    local_datastore_transcript_index,
    local_transcript_index,
)


class ViewerDataRepository:
    """Cache datastore reads needed by viewer display builders."""

    def __init__(self) -> None:
        """Initialize an empty repository."""
        self.datastore: Any | None = None
        self._datastore_transcripts: Any | None = None
        self._datastore_transcripts_loaded = False
        self._datastore_transcripts_by_gene_selection: dict[tuple[str, ...], Any] = {}
        self._proseg_transcripts: dict[str, Any | None] = {}
        self._baysor_transcripts: Any | None = None
        self._baysor_transcripts_loaded = False
        self._cellpose_boundaries: Any | None = None
        self._cellpose_boundaries_loaded = False
        self._point_indices: dict[tuple[Any, ...], Any] = {}

    def set_datastore(self, datastore: Any) -> None:
        """
        Set the active datastore and clear cached derived data.

        Parameters
        ----------
        datastore : Any
            qi2lab datastore-like object.
        """
        self.datastore = datastore
        self._datastore_transcripts = None
        self._datastore_transcripts_loaded = False
        self._datastore_transcripts_by_gene_selection.clear()
        self._proseg_transcripts.clear()
        self._baysor_transcripts = None
        self._baysor_transcripts_loaded = False
        self._cellpose_boundaries = None
        self._cellpose_boundaries_loaded = False
        self._point_indices.clear()

    def require_datastore(self) -> Any:
        """
        Return the active datastore or raise a clear error.

        Returns
        -------
        Any
            Active qi2lab datastore-like object.
        """
        if self.datastore is None:
            raise ValueError("Select a datastore first.")
        return self.datastore

    def transcript_index(
        self,
        *,
        source: str,
        shape_zyx: tuple[int, int, int],
        origin_zyx_um: Any,
        spacing_zyx_um: Any,
        tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        proseg_run_name: str | None,
        selected_genes: tuple[str, ...],
    ) -> Any:
        """
        Return a cached transcript coordinate index.

        Parameters
        ----------
        source : str
            Transcript source name.
        shape_zyx : tuple[int, int, int]
            Target Z, Y, X canvas shape.
        origin_zyx_um : Any
            Global image origin in microns.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing for local views.
        proseg_run_name : str or None
            Selected Proseg run name.
        selected_genes : tuple[str, ...]
            Transcript genes selected for display.

        Returns
        -------
        Any
            Cached transcript coordinate index.
        """
        mode = "global" if tile_transform is None else "local"
        key = (
            source,
            proseg_run_name,
            mode,
            tuple(sorted(selected_genes)),
            shape_zyx,
            self._array_key(origin_zyx_um),
            self._array_key(spacing_zyx_um),
            self._tile_transform_key(tile_transform),
        )
        if key in self._point_indices:
            return self._point_indices[key]
        table = self.transcript_table(source, proseg_run_name, selected_genes)
        index = self._build_transcript_index(
            source=source,
            table=table,
            shape_zyx=shape_zyx,
            origin_zyx_um=origin_zyx_um,
            spacing_zyx_um=spacing_zyx_um,
            tile_transform=tile_transform,
            selected_genes=selected_genes,
        )
        self._point_indices[key] = index
        return index

    def transcript_table(
        self,
        source: str,
        proseg_run_name: str | None,
        selected_genes: tuple[str, ...] = (),
    ) -> Any:
        """
        Load and cache the selected transcript table.

        Parameters
        ----------
        source : str
            Transcript source name.
        proseg_run_name : str or None
            Selected Proseg run name.
        selected_genes : tuple[str, ...], default=()
            Transcript genes selected for display.

        Returns
        -------
        Any
            Transcript table for the selected source.
        """
        datastore = self.require_datastore()
        if source == "datastore":
            selected_key = tuple(sorted(selected_genes))
            if len(selected_key) > 0:
                if selected_key not in self._datastore_transcripts_by_gene_selection:
                    self._datastore_transcripts_by_gene_selection[selected_key] = (
                        datastore.load_global_filtered_decoded_spots(
                            gene_ids=selected_key,
                            columns=("global_z", "global_y", "global_x", "gene_id"),
                        )
                    )
                return self._datastore_transcripts_by_gene_selection[selected_key]
            if not self._datastore_transcripts_loaded:
                self._datastore_transcripts = (
                    datastore.load_global_filtered_decoded_spots()
                )
                self._datastore_transcripts_loaded = True
            return self._datastore_transcripts
        if source == "proseg":
            run_name = "default" if proseg_run_name is None else proseg_run_name
            if run_name not in self._proseg_transcripts:
                self._proseg_transcripts[run_name] = (
                    datastore.load_proseg_transcripts_3d(run_name=run_name)
                )
            return self._proseg_transcripts[run_name]
        if source == "baysor":
            if not self._baysor_transcripts_loaded:
                self._baysor_transcripts = datastore.load_baysor_molecules_3d()
                self._baysor_transcripts_loaded = True
            return self._baysor_transcripts
        raise ValueError(f"Unknown transcript source: {source}")

    def tile_transform(
        self, tile: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Load one tile's global coordinate transform.

        Parameters
        ----------
        tile : str
            Tile identifier.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing, or ``None`` when unavailable.
        """
        affine, origin, spacing = self.require_datastore().load_global_coord_xforms_um(
            tile=tile
        )
        if affine is None or origin is None or spacing is None:
            return None
        return (
            np.asarray(affine, dtype=float),
            np.asarray(origin, dtype=float),
            np.asarray(spacing, dtype=float),
        )

    def cellpose_boundaries(self) -> Any:
        """
        Return cached global Cellpose boundary vectors.

        Returns
        -------
        Any
            Cellpose boundary mapping, or ``None`` when unavailable.
        """
        if self._cellpose_boundaries_loaded:
            return self._cellpose_boundaries
        datastore = self.require_datastore()
        boundaries = datastore.load_global_cellpose_roi_zip()
        if not boundaries:
            boundaries = datastore.load_global_cellpose_outlines()
        self._cellpose_boundaries = boundaries
        self._cellpose_boundaries_loaded = True
        return self._cellpose_boundaries

    @staticmethod
    def _build_transcript_index(
        *,
        source: str,
        table: Any,
        shape_zyx: tuple[int, int, int],
        origin_zyx_um: Any,
        spacing_zyx_um: Any,
        tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        selected_genes: tuple[str, ...],
    ) -> Any:
        """
        Build one global or local transcript coordinate index.

        Parameters
        ----------
        source : str
            Transcript source name.
        table : Any
            Transcript table.
        shape_zyx : tuple[int, int, int]
            Target Z, Y, X canvas shape.
        origin_zyx_um : Any
            Global image origin in microns.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing for local views.
        selected_genes : tuple[str, ...]
            Transcript genes selected for display.

        Returns
        -------
        Any
            Transcript coordinate index.
        """
        table = ViewerDataRepository._filter_table_by_genes(table, selected_genes)
        if tile_transform is None:
            if source == "datastore":
                return global_datastore_transcript_index(
                    table, shape_zyx, origin_zyx_um, spacing_zyx_um
                )
            return global_transcript_index(
                table, shape_zyx, origin_zyx_um, spacing_zyx_um
            )
        affine, origin, spacing = tile_transform
        if source == "datastore":
            return local_datastore_transcript_index(
                table, shape_zyx, affine, origin, spacing
            )
        return local_transcript_index(table, shape_zyx, affine, origin, spacing)

    @staticmethod
    def _filter_table_by_genes(table: Any, selected_genes: tuple[str, ...]) -> Any:
        """
        Return rows matching selected genes.

        Parameters
        ----------
        table : Any
            Transcript table.
        selected_genes : tuple[str, ...]
            Selected transcript genes.

        Returns
        -------
        Any
            Filtered transcript table.
        """
        if table is None or len(selected_genes) == 0:
            return table
        column = "gene_id" if "gene_id" in table.columns else "gene"
        if column not in table.columns:
            return table
        return table[table[column].astype(str).isin(selected_genes)]

    @staticmethod
    def _array_key(values: Any) -> tuple[float, ...] | None:
        """
        Return an immutable cache key for a numeric array.

        Parameters
        ----------
        values : Any
            Numeric values or ``None``.

        Returns
        -------
        tuple[float, ...] or None
            Flattened numeric cache key.
        """
        if values is None:
            return None
        return tuple(float(value) for value in np.asarray(values, dtype=float).ravel())

    @staticmethod
    def _tile_transform_key(
        tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> (
        tuple[
            tuple[float, ...] | None,
            tuple[float, ...] | None,
            tuple[float, ...] | None,
        ]
        | None
    ):
        """
        Return an immutable cache key for a tile transform.

        Parameters
        ----------
        tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing.

        Returns
        -------
        tuple[tuple[float, ...] or None, ...] or None
            Immutable tile-transform cache key.
        """
        if tile_transform is None:
            return None
        affine, origin, spacing = tile_transform
        return (
            ViewerDataRepository._array_key(affine),
            ViewerDataRepository._array_key(origin),
            ViewerDataRepository._array_key(spacing),
        )
