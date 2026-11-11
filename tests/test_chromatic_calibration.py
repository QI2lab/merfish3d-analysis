"""Bead-image calibration through the package's OME-TIFF reader."""

import numpy as np
import pytest
from tifffile import imwrite

from merfish3danalysis.utils.chromatic import load_bead_channel_stack


@pytest.mark.unit
@pytest.mark.parametrize("axes", ["CZYX", "ZCYX"])
@pytest.mark.parametrize("unit, multiplier", [("µm", 1), ("nm", 1000)])
def test_bead_stack_preserves_ome_pixels_channel_order_and_units(
    tmp_path, axes, unit, multiplier
):
    stack = np.arange(3 * 5 * 8 * 9, dtype=np.uint16).reshape(3, 5, 8, 9)
    pixels = stack if axes == "CZYX" else stack.transpose(1, 0, 2, 3)
    path = tmp_path / "beads.ome.tiff"
    names = ["01", "02", "reference"]
    wavelengths = np.array([0.52, 0.59, 0.67])
    imwrite(
        path,
        pixels,
        kind="ome",
        photometric="minisblack",
        metadata={
            "axes": axes,
            "PhysicalSizeZ": 0.315 * multiplier,
            "PhysicalSizeZUnit": unit,
            "PhysicalSizeY": 0.081 * multiplier,
            "PhysicalSizeYUnit": unit,
            "PhysicalSizeX": 0.081 * multiplier,
            "PhysicalSizeXUnit": unit,
            "Channel": {
                "Name": names,
                "EmissionWavelength": (wavelengths * multiplier).tolist(),
                "EmissionWavelengthUnit": [unit] * 3,
            },
        },
    )

    observed, metadata = load_bead_channel_stack(path)
    np.testing.assert_array_equal(observed, stack)
    np.testing.assert_allclose(metadata["voxel_size_zyx_um"], [0.315, 0.081, 0.081])
    np.testing.assert_allclose(metadata["wavelengths_um"], wavelengths)
    assert metadata["channel_names"] == names


@pytest.mark.unit
def test_bead_stack_accepts_explicit_calibration_for_plain_tiff(tmp_path):
    stack = np.arange(2 * 5 * 8 * 9, dtype=np.uint16).reshape(2, 5, 8, 9)
    path = tmp_path / "beads.tiff"
    imwrite(
        path, stack, kind="shaped", photometric="minisblack", metadata={"axes": "CZYX"}
    )

    observed, metadata = load_bead_channel_stack(
        path, voxel_size_zyx_um=(0.315, 0.081, 0.081), wavelengths_um=(0.52, 0.67)
    )
    np.testing.assert_array_equal(observed, stack)
    np.testing.assert_allclose(metadata["voxel_size_zyx_um"], [0.315, 0.081, 0.081])
    np.testing.assert_allclose(metadata["wavelengths_um"], [0.52, 0.67])
