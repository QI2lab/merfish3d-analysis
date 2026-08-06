from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, sentinel

import numpy as np
import pytest
from joblib._parallel_backends import LokyBackend

import merfish3danalysis.DataRegistration as data_registration_module
from merfish3danalysis.cli.qi2lab_microscopes import fuseall
from merfish3danalysis.DataRegistration import _direct_zarr_fusion_kwargs


def test_direct_zarr_fusion_options_match_global_fusion_parallelization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_batch = sentinel.process_batch
    monkeypatch.setattr(data_registration_module.os, "cpu_count", lambda: 3)

    options = _direct_zarr_fusion_kwargs(
        misc_utils=SimpleNamespace(process_batch_using_joblib=process_batch)
    )

    assert options["zarr_options"] == {
        "ome_zarr": True,
        "ngff_version": "0.5",
        "overwrite": True,
    }
    assert options["backend"] == "numpy"
    assert options["batch_options"]["batch_func"] is process_batch
    assert options["batch_options"]["n_batch"] == 12
    batch_kwargs = options["batch_options"]["batch_func_kwargs"]
    assert batch_kwargs["n_jobs"] == 3
    assert isinstance(batch_kwargs["backend"], LokyBackend)
    assert batch_kwargs["backend"].inner_max_num_threads == 1
    assert batch_kwargs["backend"].backend_kwargs["idle_worker_timeout"] == 86400
    assert (
        batch_kwargs["backend"].backend_kwargs["initializer"]
        is data_registration_module._configure_loky_fusion_worker
    )


def test_require_global_transforms_reports_every_missing_tile() -> None:
    identity = np.eye(4, dtype=np.float32)
    transforms = {
        "tile0000": identity,
        "tile0001": None,
        "tile0002": None,
    }
    datastore = SimpleNamespace(
        tile_ids=list(transforms),
        load_global_coord_xforms_um=lambda *, tile: (
            transforms[tile],
            None,
            None,
        ),
    )

    with pytest.raises(RuntimeError, match=r"tile0001.*tile0002"):
        fuseall._require_global_transforms(datastore)


def test_load_tile_multichannel_msim_orders_channels_and_attaches_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_affine = np.arange(16, dtype=np.float32).reshape(4, 4)
    camera_affine = np.eye(4, dtype=np.float32)
    fiducial_data = np.ones((2, 3, 4), dtype=np.uint16)
    readout_data = [
        np.full((2, 3, 4), 2, dtype=np.uint16),
        np.full((2, 3, 4), 3, dtype=np.uint16),
    ]
    read_readout_data = Mock(side_effect=readout_data)
    get_sim_from_array = Mock(return_value=sentinel.sim)
    get_msim_from_sim = Mock(return_value=sentinel.msim)
    set_affine_transform = Mock()
    monkeypatch.setattr(
        fuseall,
        "_local_fiducial_path",
        Mock(return_value=Path("/unused/fiducial.ome.zarr")),
    )
    monkeypatch.setattr(
        fuseall,
        "_lazy_zyx_data",
        Mock(return_value=fiducial_data),
    )
    monkeypatch.setattr(fuseall, "_read_readout_data", read_readout_data)

    datastore = SimpleNamespace(
        round_ids=["round001"],
        voxel_size_zyx_um=(0.4, 0.1, 0.1),
        load_local_stage_position_zyx_um=Mock(
            return_value=(np.asarray((1.234, 5.678, 9.101)), camera_affine)
        ),
        load_global_coord_xforms_um=Mock(
            return_value=(global_affine, np.zeros(3), np.ones(3))
        ),
    )
    msi_utils_module = SimpleNamespace(
        get_msim_from_sim=get_msim_from_sim,
        set_affine_transform=set_affine_transform,
    )
    si_utils_module = SimpleNamespace(get_sim_from_array=get_sim_from_array)
    open_array = Mock(return_value=sentinel.fiducial_array)

    result = fuseall._load_tile_multichannel_msim(
        datastore=datastore,
        tile_id="tile0000",
        bit_ids=["bit001", "bit002"],
        zarr_module=SimpleNamespace(open_array=open_array),
        da_module=SimpleNamespace(stack=np.stack),
        msi_utils_module=msi_utils_module,
        si_utils_module=si_utils_module,
    )

    assert result is sentinel.msim
    open_array.assert_called_once_with(Path("/unused/fiducial.ome.zarr/0"), mode="r")
    assert [item.kwargs["bit_id"] for item in read_readout_data.call_args_list] == [
        "bit001",
        "bit002",
    ]
    get_sim_from_array.assert_called_once()
    sim_call = get_sim_from_array.call_args
    np.testing.assert_array_equal(
        sim_call.args[0],
        np.stack([fiducial_data, *readout_data], axis=0),
    )
    assert sim_call.kwargs == {
        "dims": ("c", "z", "y", "x"),
        "scale": {"z": 0.4, "y": 0.1, "x": 0.1},
        "translation": {"z": 1.23, "y": 5.68, "x": 9.1},
        "affine": camera_affine,
        "transform_key": "stage_metadata",
        "c_coords": ["fiducial", "bit001", "bit002"],
        "t_coords": None,
    }
    get_msim_from_sim.assert_called_once_with(sentinel.sim, scale_factors=[])
    set_affine_transform.assert_called_once()
    transform_args = set_affine_transform.call_args
    assert transform_args.args[0] is sentinel.msim
    np.testing.assert_array_equal(transform_args.args[1], global_affine[None, ...])
    assert transform_args.kwargs == {"transform_key": "global_registered"}


def test_fuse_all_channels_writes_one_ordered_multichannel_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = np.eye(4, dtype=np.float32)
    datastore = SimpleNamespace(
        tile_ids=["tile0000", "tile0001"],
        bit_ids=["bit010", "bit002", "bit001"],
        load_global_coord_xforms_um=Mock(return_value=(identity, None, None)),
    )
    load_tile_multichannel_msim = Mock(side_effect=[sentinel.tile0, sentinel.tile1])
    direct_kwargs = {
        "zarr_options": sentinel.zarr_options,
        "batch_options": sentinel.batch_options,
        "backend": "numpy",
    }
    direct_zarr_fusion_kwargs = Mock(return_value=direct_kwargs)
    fuse = Mock(return_value=sentinel.fused_all_channels)
    write_fused_metadata = Mock()

    monkeypatch.setattr(fuseall, "qi2labDataStore", Mock(return_value=datastore))
    monkeypatch.setattr(
        fuseall,
        "_load_tile_multichannel_msim",
        load_tile_multichannel_msim,
    )
    monkeypatch.setattr(
        fuseall,
        "_direct_zarr_fusion_kwargs",
        direct_zarr_fusion_kwargs,
    )
    monkeypatch.setattr(fuseall.fusion, "fuse", fuse)
    monkeypatch.setattr(fuseall, "_write_fused_metadata", write_fused_metadata)
    monkeypatch.setattr(fuseall, "tqdm", lambda iterable, **_kwargs: iterable)

    fuseall.fuse_all_channels(tmp_path)

    assert load_tile_multichannel_msim.call_args_list == [
        call(
            datastore=datastore,
            tile_id="tile0000",
            bit_ids=["bit001", "bit002", "bit010"],
        ),
        call(
            datastore=datastore,
            tile_id="tile0001",
            bit_ids=["bit001", "bit002", "bit010"],
        ),
    ]
    direct_zarr_fusion_kwargs.assert_called_once_with(misc_utils=fuseall.misc_utils)
    fuse.assert_called_once_with(
        images=[sentinel.tile0, sentinel.tile1],
        transform_key="global_registered",
        output_zarr_url=str(tmp_path / "fused" / "fused_all_channels.ome.zarr"),
        **direct_kwargs,
    )
    write_fused_metadata.assert_called_once_with(
        fused_msim=sentinel.fused_all_channels,
        output_path=tmp_path / "fused" / "fused_all_channels.ome.zarr",
    )
