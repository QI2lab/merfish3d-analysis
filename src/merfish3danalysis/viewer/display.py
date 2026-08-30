"""Display model for datastore viewer image and sparse-overlay builds."""

from typing import Any

import numpy as np

from merfish3danalysis.viewer.models import (
    DisplayContext,
    GlobalDisplayRequest,
    LocalDisplayRequest,
    SparseLineLayer,
    SparseOverlayPayload,
    SparsePointLayer,
    TranscriptRefreshRequest,
    ViewerBuildResult,
)
from merfish3danalysis.viewer.overlays import (
    global_baysor_boundary_geometry,
    global_cell_boundary_geometry_from_source,
    load_global_image_channels,
    local_baysor_boundary_geometry,
    local_cell_boundary_geometry,
)
from merfish3danalysis.viewer.repository import ViewerDataRepository
from merfish3danalysis.viewer.warping import load_local_warped_image_channels


class ViewerDisplayModel:
    """Prepare viewer display payloads from one datastore repository."""

    def __init__(self, repository: ViewerDataRepository | None = None) -> None:
        """
        Initialize an empty display model.

        Parameters
        ----------
        repository : ViewerDataRepository or None
            Optional datastore repository.
        """
        self.repository = ViewerDataRepository() if repository is None else repository

    def set_datastore(self, datastore: Any) -> None:
        """
        Set the active datastore and clear derived caches.

        Parameters
        ----------
        datastore : Any
            qi2lab datastore-like object.
        """
        self.repository.set_datastore(datastore)

    def build_global(self, request: GlobalDisplayRequest) -> ViewerBuildResult:
        """
        Build a global fused display result.

        Parameters
        ----------
        request : GlobalDisplayRequest
            Global display selection state.

        Returns
        -------
        ViewerBuildResult
            Prepared global image and sparse overlay payload.
        """
        datastore = self.repository.require_datastore()
        global_stack = load_global_image_channels(
            datastore,
            include_fused_image=request.include_fused_image,
            include_segmentation=request.include_segmentation,
            max_project_fused_image=request.max_project,
        )
        stack = global_stack.stack
        shape_zyx = tuple(int(value) for value in stack.data.shape[1:])
        sparse_payload = self._global_sparse_payload(
            request,
            shape_zyx=shape_zyx,
            source_shape_zyx=global_stack.full_shape_zyx,
            origin_zyx_um=global_stack.origin_zyx_um,
            spacing_zyx_um=global_stack.spacing_zyx_um,
        )
        context = DisplayContext(
            mode="global",
            base_stack=stack,
            base_sparse_lines=sparse_payload.lines,
            shape_zyx=shape_zyx,
            origin_zyx_um=global_stack.origin_zyx_um,
            spacing_zyx_um=global_stack.spacing_zyx_um,
            full_shape_zyx=global_stack.full_shape_zyx,
        )
        return ViewerBuildResult(
            stack=stack,
            spacing_zyx_um=global_stack.spacing_zyx_um,
            origin_zyx_um=global_stack.origin_zyx_um,
            context=context,
            status=self._status(stack.labels, sparse_payload),
            sparse_payload=sparse_payload,
        )

    def build_local(self, request: LocalDisplayRequest) -> ViewerBuildResult:
        """
        Build a local tile display result.

        Parameters
        ----------
        request : LocalDisplayRequest
            Local tile display selection state.

        Returns
        -------
        ViewerBuildResult
            Prepared local image and sparse overlay payload.
        """
        datastore = self.repository.require_datastore()
        display = load_local_warped_image_channels(
            datastore,
            tile=request.tile,
            fiducial_round_ids=list(request.fiducial_rounds),
            fiducial_sources=list(request.fiducial_sources),
            bit_ids=list(request.bit_ids),
            bit_sources=list(request.bit_sources),
            options=request.warp_options,
            gpu_id=request.gpu_id,
        )
        stack = display.stack
        shape_zyx = tuple(int(value) for value in stack.data.shape[1:])
        tile_transform = self.repository.tile_transform(request.tile)
        sparse_payload = self._local_sparse_payload(
            request,
            shape_zyx=shape_zyx,
            tile_transform=tile_transform,
        )
        context = DisplayContext(
            mode="local",
            base_stack=stack,
            base_sparse_lines=sparse_payload.lines,
            shape_zyx=shape_zyx,
            tile=request.tile,
            spacing_zyx_um=np.asarray(datastore.voxel_size_zyx_um),
            tile_transform=tile_transform,
        )
        status = self._status(stack.labels, sparse_payload)
        if display.warnings:
            status += "\nWarnings: " + "; ".join(display.warnings)
        return ViewerBuildResult(
            stack=stack,
            spacing_zyx_um=np.asarray(datastore.voxel_size_zyx_um),
            origin_zyx_um=None,
            context=context,
            status=status,
            sparse_payload=sparse_payload,
        )

    def build_transcript_refresh(
        self,
        context: DisplayContext,
        request: TranscriptRefreshRequest,
    ) -> ViewerBuildResult:
        """
        Rebuild transcript sparse overlays for an existing display.

        Parameters
        ----------
        context : DisplayContext
            Current display context.
        request : TranscriptRefreshRequest
            Transcript overlay selection state.

        Returns
        -------
        ViewerBuildResult
            Overlay-only refresh result.
        """
        stack = context.base_stack
        shape_zyx = tuple(int(value) for value in context.shape_zyx)
        if context.mode == "global":
            transcript_payload = self._transcript_payload(
                source=request.source,
                shape_zyx=shape_zyx,
                origin_zyx_um=context.origin_zyx_um,
                spacing_zyx_um=context.spacing_zyx_um,
                tile_transform=None,
                selected_genes=request.selected_genes,
                marker_radius=request.marker_radius,
                proseg_run_name=request.proseg_run_name,
            )
            spacing_zyx_um = context.spacing_zyx_um
            origin_zyx_um = context.origin_zyx_um
        else:
            tile_transform = context.tile_transform
            transcript_payload = self._transcript_payload(
                source=request.source,
                shape_zyx=shape_zyx,
                origin_zyx_um=None,
                spacing_zyx_um=context.spacing_zyx_um,
                tile_transform=tile_transform,
                selected_genes=request.selected_genes,
                marker_radius=request.marker_radius,
                proseg_run_name=request.proseg_run_name,
            )
            spacing_zyx_um = context.spacing_zyx_um
            origin_zyx_um = None
        sparse_payload = SparseOverlayPayload(
            points=transcript_payload.points,
            lines=context.base_sparse_lines,
        )
        return ViewerBuildResult(
            stack=stack,
            spacing_zyx_um=np.asarray(spacing_zyx_um),
            origin_zyx_um=origin_zyx_um,
            context=context.as_refresh(),
            status=self._status(stack.labels, sparse_payload),
            sparse_payload=sparse_payload,
        )

    def _global_sparse_payload(
        self,
        request: GlobalDisplayRequest,
        *,
        shape_zyx: tuple[int, int, int],
        source_shape_zyx: tuple[int, int, int],
        origin_zyx_um: np.ndarray,
        spacing_zyx_um: np.ndarray,
    ) -> SparseOverlayPayload:
        """
        Build sparse overlays for a global display.

        Parameters
        ----------
        request : GlobalDisplayRequest
            Global display selection state.
        shape_zyx : tuple[int, int, int]
            Global canvas shape.
        source_shape_zyx : tuple[int, int, int]
            Full source Z, Y, X shape before max projection.
        origin_zyx_um : numpy.ndarray
            Global image origin in microns.
        spacing_zyx_um : numpy.ndarray
            Z, Y, X voxel spacing in microns.

        Returns
        -------
        SparseOverlayPayload
            Sparse overlay payload.
        """
        lines: list[SparseLineLayer] = []
        datastore = self.repository.require_datastore()
        if request.include_cell_boundaries:
            geometry = global_cell_boundary_geometry_from_source(
                self.repository.cellpose_boundaries(),
                shape_zyx,
                origin_zyx_um,
                spacing_zyx_um,
                line_thickness=5,
            )
            self._append_line_layer(
                lines, geometry, spacing_zyx_um, "global Cellpose cell boundaries"
            )
        if request.include_proseg_boundaries and request.proseg_run_name is not None:
            geometry = global_cell_boundary_geometry_from_source(
                datastore.load_proseg_cell_polygons_3d(
                    run_name=request.proseg_run_name
                ),
                shape_zyx,
                origin_zyx_um,
                spacing_zyx_um,
                line_thickness=5,
            )
            self._append_line_layer(
                lines,
                geometry,
                spacing_zyx_um,
                f"global Proseg cell boundaries {request.proseg_run_name}",
            )
        if request.include_baysor_boundaries:
            geometry = global_baysor_boundary_geometry(
                datastore.load_baysor_cell_boundaries_3d(),
                shape_zyx,
                origin_zyx_um,
                spacing_zyx_um,
                line_thickness=1,
                source_shape_zyx=source_shape_zyx,
            )
            self._append_line_layer(
                lines, geometry, spacing_zyx_um, "global Baysor cell boundaries"
            )
        transcript_payload = self._transcript_payload(
            source=request.transcript_source,
            shape_zyx=shape_zyx,
            origin_zyx_um=origin_zyx_um,
            spacing_zyx_um=spacing_zyx_um,
            tile_transform=None,
            selected_genes=request.selected_genes,
            marker_radius=request.marker_radius,
            proseg_run_name=request.proseg_run_name,
        )
        return SparseOverlayPayload(
            points=transcript_payload.points,
            lines=(*lines, *transcript_payload.lines),
        )

    def _local_sparse_payload(
        self,
        request: LocalDisplayRequest,
        *,
        shape_zyx: tuple[int, int, int],
        tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> SparseOverlayPayload:
        """
        Build sparse overlays for a local display.

        Parameters
        ----------
        request : LocalDisplayRequest
            Local tile display selection state.
        shape_zyx : tuple[int, int, int]
            Local canvas shape.
        tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing for global overlays.

        Returns
        -------
        SparseOverlayPayload
            Sparse overlay payload.
        """
        lines: list[SparseLineLayer] = []
        datastore = self.repository.require_datastore()
        spacing_zyx_um = np.asarray(datastore.voxel_size_zyx_um)
        if tile_transform is not None:
            affine, origin, spacing = tile_transform
            if request.include_cell_boundaries:
                geometry = local_cell_boundary_geometry(
                    self.repository.cellpose_boundaries(),
                    shape_zyx,
                    affine,
                    origin,
                    spacing,
                    line_thickness=5,
                )
                self._append_line_layer(
                    lines, geometry, spacing_zyx_um, "Cellpose cell boundaries"
                )
            if (
                request.include_proseg_boundaries
                and request.proseg_run_name is not None
            ):
                geometry = local_cell_boundary_geometry(
                    datastore.load_proseg_cell_polygons_3d(
                        run_name=request.proseg_run_name
                    ),
                    shape_zyx,
                    affine,
                    origin,
                    spacing,
                    line_thickness=5,
                )
                self._append_line_layer(
                    lines,
                    geometry,
                    spacing_zyx_um,
                    f"Proseg cell boundaries {request.proseg_run_name}",
                )
            if request.include_baysor_boundaries:
                geometry = local_baysor_boundary_geometry(
                    datastore.load_baysor_cell_boundaries_3d(),
                    shape_zyx,
                    affine,
                    origin,
                    spacing,
                    line_thickness=1,
                )
                self._append_line_layer(
                    lines, geometry, spacing_zyx_um, "Baysor cell boundaries"
                )
        transcript_payload = self._transcript_payload(
            source=request.transcript_source if tile_transform is not None else None,
            shape_zyx=shape_zyx,
            origin_zyx_um=None,
            spacing_zyx_um=spacing_zyx_um,
            tile_transform=tile_transform,
            selected_genes=request.selected_genes,
            marker_radius=request.marker_radius,
            proseg_run_name=request.proseg_run_name,
        )
        return SparseOverlayPayload(
            points=transcript_payload.points,
            lines=(*lines, *transcript_payload.lines),
        )

    def _transcript_payload(
        self,
        *,
        source: str | None,
        shape_zyx: tuple[int, int, int],
        origin_zyx_um: Any,
        spacing_zyx_um: Any,
        tile_transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        selected_genes: tuple[str, ...],
        marker_radius: int,
        proseg_run_name: str | None,
    ) -> SparseOverlayPayload:
        """
        Build one transcript point layer for the selected source.

        Parameters
        ----------
        source : str or None
            Transcript source name.
        shape_zyx : tuple[int, int, int]
            Target Z, Y, X canvas shape.
        origin_zyx_um : Any
            Global image origin in microns.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        tile_transform : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
            Tile affine, origin, and spacing for local views.
        selected_genes : tuple[str, ...]
            Selected transcript genes.
        marker_radius : int
            Marker radius in display pixels.
        proseg_run_name : str or None
            Selected Proseg run name.

        Returns
        -------
        SparseOverlayPayload
            Transcript point payload.
        """
        if source is None or len(selected_genes) == 0:
            return SparseOverlayPayload()
        index = self.repository.transcript_index(
            source=source,
            shape_zyx=shape_zyx,
            origin_zyx_um=origin_zyx_um,
            spacing_zyx_um=spacing_zyx_um,
            tile_transform=tile_transform,
            proseg_run_name=proseg_run_name,
            selected_genes=selected_genes,
        )
        return SparseOverlayPayload(
            points=(
                SparsePointLayer(
                    index=index,
                    selected_genes=selected_genes,
                    marker_size=marker_radius,
                    spacing_zyx_um=self._spacing_tuple(spacing_zyx_um),
                    label=f"{source} transcripts",
                ),
            )
        )

    @staticmethod
    def _append_line_layer(
        layers: list[SparseLineLayer],
        geometry: Any,
        spacing_zyx_um: Any,
        label: str,
    ) -> None:
        """
        Append sparse line geometry when available.

        Parameters
        ----------
        layers : list[SparseLineLayer]
            Destination layer list.
        geometry : Any
            Sparse geometry object or ``None``.
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.
        label : str
            Display label for the layer.
        """
        if geometry is None:
            return
        polylines = getattr(geometry, "polylines", None)
        if polylines is None:
            polylines = getattr(geometry, "polylines_yx", ())
        layers.append(
            SparseLineLayer(
                polylines=polylines,
                shape=geometry.shape,
                width=geometry.line_thickness,
                color="white",
                spacing_zyx_um=ViewerDisplayModel._spacing_tuple(spacing_zyx_um),
                label=label,
                z_aware=hasattr(geometry, "z_polylines_by_index"),
                z_polylines_by_index=getattr(geometry, "z_polylines_by_index", None),
                max_polylines_yx=getattr(geometry, "max_polylines_yx", ()),
            )
        )

    @staticmethod
    def _status(labels: list[str], sparse_payload: SparseOverlayPayload) -> str:
        """
        Return a compact display status string.

        Parameters
        ----------
        labels : list[str]
            Image channel labels.
        sparse_payload : SparseOverlayPayload
            Sparse overlay payload.

        Returns
        -------
        str
            Human-readable display status.
        """
        return "Displayed: " + ", ".join([*labels, *sparse_payload.labels])

    @staticmethod
    def _spacing_tuple(spacing_zyx_um: Any) -> tuple[float, float, float]:
        """
        Return Z, Y, X spacing as a float tuple.

        Parameters
        ----------
        spacing_zyx_um : Any
            Z, Y, X voxel spacing in microns.

        Returns
        -------
        tuple[float, float, float]
            Spacing as Python floats.
        """
        spacing = np.asarray(spacing_zyx_um, dtype=float)
        return float(spacing[0]), float(spacing[1]), float(spacing[2])
