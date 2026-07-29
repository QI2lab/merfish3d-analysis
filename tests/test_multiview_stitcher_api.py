from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np

from merfish3danalysis.DataRegistration import _read_fiducial_sim


def test_read_fiducial_sim_uses_zarr_array_backend() -> None:
    sim_on_disk = SimpleNamespace(
        data=sentinel.image_data,
        dims=("z", "y", "x"),
        coords={},
    )
    read_sim = Mock(return_value=sim_on_disk)
    get_sim_from_array = Mock(return_value=sentinel.spatial_image)
    affine_zyx_px = np.eye(4)

    result = _read_fiducial_sim(
        input_path=Path("/unused/input.ome.zarr"),
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 3.0, "y": 4.0, "x": 5.0},
        affine_zyx_px=affine_zyx_px,
        transform_key="stage_metadata",
        ngff_utils=SimpleNamespace(read_sim_from_ome_zarr=read_sim),
        si_utils=SimpleNamespace(get_sim_from_array=get_sim_from_array),
    )

    assert result is sentinel.spatial_image
    read_sim.assert_called_once_with(
        Path("/unused/input.ome.zarr"),
        resolution_level=0,
        transform_key="stage_metadata",
        array_backend="zarr",
    )
    get_sim_from_array.assert_called_once_with(
        sentinel.image_data,
        dims=("z", "y", "x"),
        scale={"z": 2.0, "y": 0.5, "x": 0.5},
        translation={"z": 3.0, "y": 4.0, "x": 5.0},
        affine=affine_zyx_px,
        transform_key="stage_metadata",
        c_coords=None,
        t_coords=None,
    )
