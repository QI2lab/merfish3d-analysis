"""Data models and coordinate helpers for the datastore viewer."""

from dataclasses import dataclass
from typing import Any

import numpy as np


class ChannelStack:
    """Channel-stacked image data and labels for display."""

    data: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class GlobalChannelStack:
    """Global channel stack with micron coordinate metadata."""

    stack: ChannelStack
    origin_zyx_um: np.ndarray
    spacing_zyx_um: np.ndarray


@dataclass(frozen=True)
class ProsegRun:
    """One detected Proseg 3D output run."""

    name: str


@dataclass(frozen=True)
class ViewerDisplay:
    """Display payload and warning messages produced by viewer loaders."""

    stack: ChannelStack
    warnings: list[str]


@dataclass(frozen=True)
class ViewerBuildResult:
    """Prepared viewer data produced off the Qt UI thread."""

    stack: ChannelStack
    spacing_zyx_um: np.ndarray
    origin_zyx_um: np.ndarray | None
    context: dict[str, Any]
    status: str
    decoded_spots: Any | None = None
    proseg_run_name: str | None = None
    proseg_transcripts: Any | None = None


@dataclass(frozen=True)
class WarpChainOptions:
    """Viewer-selected local warp-chain components."""

    chromatic: bool = True
    stage_affine: bool = True
    sofima: bool = True

    @property
    def is_native(self) -> bool:
        """
        Return whether all transform components are disabled.

        Returns
        -------
        bool
            True when the viewer should display stored native coordinates.
        """

        return not (self.chromatic or self.stage_affine or self.sofima)


def warp_chain_label(options: WarpChainOptions) -> str:
    """
    Return a compact viewer label for selected warp-chain components.

    Parameters
    ----------
    options : WarpChainOptions
        Selected local viewer transform components.

    Returns
    -------
    str
        Human-readable warp-chain label.
    """

    pieces: list[str] = []
    if options.stage_affine:
        pieces.append("affine")
    if options.chromatic:
        pieces.append("chromatic")
    if options.sofima:
        pieces.append("sofima")
    return "+".join(pieces) if pieces else "native"


def compose_viewer_warp_transform_zyx_um(
    *,
    round_transform_zyx_um: np.ndarray | None,
    chromatic_transform_zyx_um: np.ndarray | None,
) -> np.ndarray:
    """
    Compose optional affine pieces for local viewer display.

    Parameters
    ----------
    round_transform_zyx_um : numpy.ndarray or None
        Physical round/stage affine. ``None`` means identity.
    chromatic_transform_zyx_um : numpy.ndarray or None
        Physical chromatic affine. ``None`` means identity.

    Returns
    -------
    numpy.ndarray
        Physical transform mapping reference-frame output coordinates into
        native moving-image coordinates.
    """

    round_transform = (
        np.eye(4, dtype=np.float32)
        if round_transform_zyx_um is None
        else np.asarray(round_transform_zyx_um, dtype=np.float32)
    )
    chromatic_transform = (
        np.eye(4, dtype=np.float32)
        if chromatic_transform_zyx_um is None
        else np.asarray(chromatic_transform_zyx_um, dtype=np.float32)
    )
    return np.linalg.inv(chromatic_transform) @ round_transform


def stack_with_micron_coords(
    stack: ChannelStack,
    voxel_size_zyx_um: Any,
    origin_zyx_um: Any | None = None,
) -> Any:
    """
    Attach zyx micron coordinates to a channel stack for ndv display.

    Parameters
    ----------
    stack : ChannelStack
        stack for this viewer operation.
    voxel_size_zyx_um : Any
        voxel_size_zyx_um for this viewer operation.
    origin_zyx_um : Any | None
        origin_zyx_um for this viewer operation.

    Returns
    -------
    Any
        Computed viewer result.
    """

    import xarray as xr

    data = stack.data.astype(np.float32, copy=False)
    voxel = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_zyx_um is None
        else np.asarray(origin_zyx_um, dtype=np.float32)
    )
    if data.ndim != 4 or voxel.shape[0] != 3:
        raise ValueError("Expected channel stack shape (c, z, y, x).")
    if origin.shape[0] != 3:
        raise ValueError("Expected origin shape (3,).")

    return xr.DataArray(
        data,
        dims=("c", "z_um", "y_um", "x_um"),
        coords={
            "c": np.arange(data.shape[0]),
            "z_um": origin[0] + np.arange(data.shape[1], dtype=np.float32) * voxel[0],
            "y_um": origin[1] + np.arange(data.shape[2], dtype=np.float32) * voxel[1],
            "x_um": origin[2] + np.arange(data.shape[3], dtype=np.float32) * voxel[2],
        },
        attrs={"z_spacing_um": float(voxel[0])},
    )
