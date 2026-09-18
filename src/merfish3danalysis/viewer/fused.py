"""Lazy NDV display of the saved fused-channel images."""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from yaozarrs import open_group

from merfish3danalysis.utils.dataio import resolve_datastore_path


def load_fused_channels(root_path: Path) -> list[xr.DataArray]:
    """Open every fused store lazily with its saved axes and physical coordinates.

    Stores sharing a spatial grid are combined into one channel stack. Different
    grids remain separate so that viewing never resamples saved scientific data.
    """
    datastore_path = resolve_datastore_path(root_path)
    stacks = []
    for path in sorted((datastore_path / "fused").glob("*.ome.zarr")):
        group = open_group(path)
        multiscale = group.attrs["ome"]["multiscales"][0]
        dataset = multiscale["datasets"][0]
        dims = [axis["name"] for axis in multiscale["axes"]]
        array = da.from_zarr(group[dataset["path"]].to_zarr_python())
        scale = np.ones(array.ndim)
        translation = np.zeros(array.ndim)
        for transform in dataset.get("coordinateTransformations", []) + multiscale.get(
            "coordinateTransformations", []
        ):
            if transform["type"] == "scale":
                factor = np.asarray(transform["scale"])
                scale *= factor
                translation *= factor
            elif transform["type"] == "translation":
                translation += np.asarray(transform["translation"])
        data = xr.DataArray(
            array,
            dims=dims,
            coords={
                dim: np.arange(size) * scale[index] + translation[index]
                for index, (dim, size) in enumerate(zip(dims, array.shape, strict=True))
            },
        )
        if "t" in data.dims and data.sizes["t"] == 1:
            data = data.isel(t=0, drop=True)
        if "c" not in data.dims:
            data = data.expand_dims(c=[0])
        channel_names = group.attrs.get("channel_names", [])
        labels = [
            f"{path.name.removesuffix('.ome.zarr')}: {channel_names[index] if index < len(channel_names) else index}"
            for index in range(data.sizes["c"])
        ]
        data = data.assign_coords(c=labels)
        for index, stack in enumerate(stacks):
            if set(data.dims) == set(stack.dims) and all(
                data.coords[dim].equals(stack.coords[dim])
                for dim in data.dims
                if dim != "c"
            ):
                stacks[index] = xr.concat([stack, data], dim="c", join="exact")
                break
        else:
            stacks.append(data)
    if not stacks:
        raise FileNotFoundError(
            f"No fused OME-Zarr images in {datastore_path / 'fused'}"
        )
    return stacks


def view_fused_channels(root_path: Path) -> None:
    """Display all saved fused channels; use separate windows for different grids."""
    import os

    import ndv
    from ndv.models import ArrayDisplayModel, LUTModel
    from qtpy import QtWidgets

    from merfish3danalysis.viewer.ndv import (
        apply_lut_channel_labels,
    )

    stacks = load_fused_channels(root_path)
    os.environ["NDV_GUI_FRONTEND"] = "qt"
    os.environ["NDV_CANVAS_BACKEND"] = "vispy"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    colors = [
        "cmap:white",
        "cmap:magenta",
        "cmap:cyan",
        "cmap:red",
        "cmap:yellow",
        "cmasher:cosmic",
        "cmasher:dusk",
        "cmasher:eclipse",
        "cmasher:emerald",
        "chrisluts:BOP_Orange",
        "cmasher:sapphire",
        "chrisluts:BOP_Blue",
        "cmap:magenta",
        "cmap:cyan",
        "cmap:red",
        "cmap:yellow",
        "cmasher:cosmic",
    ]
    viewers = []
    for data in stacks:
        labels = list(data.coords["c"].values)
        luts = {
            index: LUTModel(
                cmap=colors[index % len(colors)],
                clims=(0, 1000) if index == 0 else (10, 500),
            )
            for index in range(len(labels))
        }
        viewer = ndv.ArrayViewer(
            data,
            display_model=ArrayDisplayModel(
                channel_axis="c",
                channel_mode="composite",
                visible_axes=("y", "x"),
                luts=luts,
            ),
        )
        apply_lut_channel_labels(viewer, labels)
        viewer.widget().setWindowTitle("Fused channels")
        viewer.widget().show()
        viewers.append(viewer)
    try:
        app.exec()
    finally:
        for viewer in viewers:
            viewer.widget().close()
