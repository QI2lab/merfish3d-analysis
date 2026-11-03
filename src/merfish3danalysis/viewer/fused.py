"""Lazy NDV display of the saved fused-channel images."""

from pathlib import Path

import numpy as np
import tensorstore as ts

from merfish3danalysis.qi2labDataStore import qi2labDataStore
from merfish3danalysis.utils.dataio import resolve_datastore_path
from merfish3danalysis.viewer.models import LazyGlobalChannelData


def load_fused_channels(root_path: Path) -> list[LazyGlobalChannelData]:
    """Open every fused store lazily with its saved axes and physical coordinates.

    Stores sharing a spatial grid are combined into one channel stack. Different
    grids remain separate so that viewing never resamples saved scientific data.
    """
    datastore_path = resolve_datastore_path(root_path)
    datastore = qi2labDataStore(datastore_path, validate=False)
    stacks = []
    for fusion_type in ("fiducial", "all_channels"):
        product = datastore.load_global_fused_image(fusion_type, return_future=None)
        if product is None:
            continue
        array, metadata = product
        # Library fusion outputs are TCZYX; the datastore setter also accepts ZYX.
        if array.ndim == 5:
            array = array[0]
        elif array.ndim == 3:
            array = ts.stack([array], axis=0)
        array = array[ts.d[:].label[["c", "z", "y", "x"]]]
        origin = metadata["origin_zyx_um"]
        spacing = metadata["spacing_zyx_um"]
        channel_names = metadata.get("channel_names", [])
        image_name = (
            f"fused_{datastore.fiducial_folder_name}_zyx"
            if fusion_type == "fiducial"
            else "full_dataset"
            if channel_names
            else "fused_all_channels_zyx"
        )
        labels = [
            f"{image_name}: {channel_names[index] if index < len(channel_names) else index}"
            for index in range(array.shape[0])
        ]
        coords = {
            "c": labels,
            **{
                dim: origin[index] + np.arange(array.shape[index + 1]) * spacing[index]
                for index, dim in enumerate("zyx")
            },
        }
        for index, (stack, stack_coords) in enumerate(stacks):
            if all(np.array_equal(coords[dim], stack_coords[dim]) for dim in "zyx"):
                dtype = np.result_type(stack.dtype.numpy_dtype, array.dtype.numpy_dtype)
                combined = ts.concat(
                    [ts.cast(stack, dtype), ts.cast(array, dtype)], axis=0
                )
                coords["c"] = [*stack_coords["c"], *labels]
                stacks[index] = (combined, coords)
                break
        else:
            stacks.append((array, coords))
    if not stacks:
        raise FileNotFoundError(f"No fused OME-Zarr images in {datastore_path}")
    return [
        LazyGlobalChannelData(
            fused_zyx=array[0],
            image_channels=tuple(array[index] for index in range(1, array.shape[0])),
            coords=coords,
        )
        for array, coords in stacks
    ]


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
        labels = list(data.coords["c"])
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
