import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, sentinel

import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import fusion, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import fuseall


def test_multiview_fusion_applies_channel_dependent_affines() -> None:
    image = np.zeros((2, 1, 3, 6), dtype=np.uint16)
    image[0, 0, 1, 2] = 100
    image[1, 0, 1, 3] = 100
    sim = si_utils.get_sim_from_array(
        image,
        dims=("c", "z", "y", "x"),
        scale={"z": 1.0, "y": 1.0, "x": 1.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        affine=np.eye(4),
        transform_key="stage_metadata",
        c_coords=["fiducial", "bit001"],
        t_coords=None,
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    transforms[1, 2, 3] = -1.0
    msi_utils.set_affine_transform(
        msim,
        xr.DataArray(
            transforms,
            dims=("c", "x_in", "x_out"),
            coords={
                "c": ["fiducial", "bit001"],
                "x_in": ["z", "y", "x", "1"],
                "x_out": ["z", "y", "x", "1"],
            },
        ),
        transform_key="global_registered",
    )

    fused = fusion.fuse(images=[msim], transform_key="global_registered")
    fused_sim = msi_utils.get_sim_from_msim(fused)
    fused_data = np.asarray(fused_sim.data.compute())[0]

    assert list(fused_sim.coords["c"].values) == ["fiducial", "bit001"]
    fiducial_peak = np.unravel_index(np.argmax(fused_data[0]), fused_data[0].shape)
    bit_peak = np.unravel_index(np.argmax(fused_data[1]), fused_data[1].shape)
    assert bit_peak == fiducial_peak


def test_channel_global_transforms_follow_repository_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tile_position = np.asarray((2.0, 10.0, 20.0), dtype=np.float32)
    stage_camera = np.eye(4, dtype=np.float32)
    stage_camera[1, 1] = -1.0
    refinement = np.eye(4, dtype=np.float32)
    refinement[2, 3] = 4.0
    round_bit001 = np.eye(4, dtype=np.float32)
    round_bit001[1, 3] = 1.5
    round_bit002 = np.eye(4, dtype=np.float32)
    round_bit002[2, 3] = -2.5
    chromatic_bit001 = np.eye(4, dtype=np.float32)
    chromatic_bit001[2, 3] = 0.25
    chromatic_bit002 = np.eye(4, dtype=np.float32)
    chromatic_bit002[1, 3] = -0.5
    load_round = Mock(
        side_effect=[
            ("round002", round_bit001),
            ("round003", round_bit002),
        ]
    )
    monkeypatch.setattr(fuseall, "load_bit_round_transform_zyx_um", load_round)
    datastore = SimpleNamespace(
        load_local_wavelengths_um=Mock(side_effect=[(0.48, 0.52), (0.58, 0.65)]),
        load_chromatic_affine_transform_zyx_um=Mock(
            side_effect=[chromatic_bit001, chromatic_bit002]
        ),
    )

    transforms = fuseall._channel_global_transforms_zyx_um(
        datastore=datastore,
        tile_id="tile0000",
        bit_ids=["bit001", "bit002"],
        tile_position_zyx_um=tile_position,
        stage_camera_zyx_um=stage_camera,
        global_refinement_zyx_um=refinement,
    )

    tile_origin = np.eye(4, dtype=np.float32)
    tile_origin[:3, 3] = tile_position
    inverse_tile_origin = np.linalg.inv(tile_origin)
    expected = np.stack(
        [
            refinement @ stage_camera,
            refinement
            @ stage_camera
            @ tile_origin
            @ np.linalg.inv(round_bit001)
            @ chromatic_bit001
            @ inverse_tile_origin,
            refinement
            @ stage_camera
            @ tile_origin
            @ np.linalg.inv(round_bit002)
            @ chromatic_bit002
            @ inverse_tile_origin,
        ]
    )
    np.testing.assert_allclose(transforms, expected, atol=1e-6)
    assert [item.kwargs["bit_id"] for item in load_round.call_args_list] == [
        "bit001",
        "bit002",
    ]


def test_load_tile_multichannel_msim_opens_zarr_inputs_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_camera = np.eye(4, dtype=np.float32)
    stage_transform = xr.DataArray(
        stage_camera,
        dims=("x_in", "x_out"),
        coords={
            "x_in": ["z", "y", "x", "1"],
            "x_out": ["z", "y", "x", "1"],
        },
    )
    channel_transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    input_paths = [
        tmp_path / "fiducial.ome.zarr",
        tmp_path / "bit001.ome.zarr",
        tmp_path / "bit002.ome.zarr",
    ]
    for input_path in input_paths:
        input_path.mkdir()
    datastore = SimpleNamespace(
        round_ids=["round001"],
        fiducial_folder_name="fiducial",
        voxel_size_zyx_um=(
            0.31999999999993634,
            0.09799999999995634,
            0.098,
        ),
        load_local_stage_position_zyx_um=Mock(
            return_value=(np.asarray((1.234, 5.678, 9.101)), stage_camera)
        ),
        load_global_coord_xforms_um=Mock(
            return_value=(np.eye(4), np.zeros(3), np.ones(3))
        ),
    )
    raw_sims = [Mock(), Mock(), Mock()]
    channel_sims = [sentinel.fiducial_sim, sentinel.bit001_sim, sentinel.bit002_sim]
    for raw_sim, channel_sim in zip(raw_sims, channel_sims, strict=True):
        raw_sim.assign_coords.return_value = channel_sim
    read_sim = Mock(side_effect=raw_sims)
    concat = Mock(return_value=sentinel.multichannel_sim)
    get_msim_from_sim = Mock(return_value=sentinel.msim)
    get_transform_from_msim = Mock(return_value=stage_transform)
    set_affine_transform = Mock()
    channel_global_transforms = Mock(return_value=channel_transforms)
    monkeypatch.setattr(
        fuseall,
        "_channel_global_transforms_zyx_um",
        channel_global_transforms,
    )
    local_fiducial_path = Mock(return_value=input_paths[0])
    local_readout_path = Mock(side_effect=input_paths[1:])
    monkeypatch.setattr(fuseall, "_local_fiducial_path", local_fiducial_path)
    monkeypatch.setattr(fuseall, "_local_readout_path", local_readout_path)
    monkeypatch.setattr(fuseall, "_read_fiducial_sim", read_sim)

    result = fuseall._load_tile_multichannel_msim(
        datastore=datastore,
        tile_id="tile0000",
        bit_ids=["bit001", "bit002"],
        zarr_module=sentinel.zarr,
        msi_utils_module=SimpleNamespace(
            get_msim_from_sim=get_msim_from_sim,
            get_transform_from_msim=get_transform_from_msim,
            set_affine_transform=set_affine_transform,
        ),
        si_utils_module=SimpleNamespace(
            concat=concat,
        ),
    )

    assert result is sentinel.msim
    local_fiducial_path.assert_called_once_with(datastore, "tile0000", "round001")
    assert local_readout_path.call_args_list == [
        call(datastore, "tile0000", "bit001"),
        call(datastore, "tile0000", "bit002"),
    ]
    expected_common = {
        "scale": {"z": 0.32, "y": 0.098, "x": 0.098},
        "translation": {"z": 1.23, "y": 5.68, "x": 9.1},
        "affine_zyx_px": stage_camera,
        "transform_key": "stage_metadata",
        "zarr_module": sentinel.zarr,
    }
    for read_call, input_path in zip(read_sim.call_args_list, input_paths, strict=True):
        assert read_call.kwargs["input_path"] == input_path
        assert read_call.kwargs["scale"] == expected_common["scale"]
        assert read_call.kwargs["translation"] == expected_common["translation"]
        np.testing.assert_array_equal(
            read_call.kwargs["affine_zyx_px"], expected_common["affine_zyx_px"]
        )
        assert read_call.kwargs["transform_key"] == "stage_metadata"
        assert read_call.kwargs["zarr_module"] is sentinel.zarr
    for raw_sim, channel_id in zip(
        raw_sims, ["fiducial", "bit001", "bit002"], strict=True
    ):
        raw_sim.assign_coords.assert_called_once_with(c=[channel_id])
    concat.assert_called_once_with(channel_sims, dim="c")
    get_msim_from_sim.assert_called_once_with(
        sentinel.multichannel_sim, scale_factors=[]
    )
    get_transform_from_msim.assert_called_once_with(
        sentinel.msim,
        transform_key="stage_metadata",
    )
    set_call = set_affine_transform.call_args
    transform_data = set_call.args[1]
    assert transform_data.dims == ("c", "x_in", "x_out")
    assert list(transform_data.coords["c"].values) == [
        "fiducial",
        "bit001",
        "bit002",
    ]
    np.testing.assert_array_equal(transform_data, channel_transforms)
    assert set_call.kwargs == {"transform_key": "global_registered"}


@pytest.mark.parametrize("write_ome_tiffs", [False, True])
def test_fuse_all_channels_writes_one_cpu_multichannel_ome_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_ome_tiffs: bool,
) -> None:
    fused_root = tmp_path / "qi2labdatastore" / "fused"
    fused_root.mkdir(parents=True)
    datastore = SimpleNamespace(
        tile_ids=["tile0000", "tile0001"],
        bit_ids=["bit010", "bit002", "bit001"],
        fiducial_folder_name="fiducial",
        voxel_size_zyx_um=(
            0.31999999999993634,
            0.09799999999995634,
            0.098,
        ),
        _fused_root_path=fused_root,
        _image_store_path=fuseall.qi2labDataStore._image_store_path,
        _write_extra_attributes=Mock(),
    )
    load_tile = Mock(side_effect=[sentinel.tile0, sentinel.tile1])
    fusion_options = {
        "zarr_options": {
            "ome_zarr": True,
            "ngff_version": "0.5",
            "overwrite": True,
        },
        "batch_options": sentinel.batch_options,
        "backend": "numpy",
    }
    direct_options = Mock(return_value=fusion_options)
    fuse = Mock(return_value=sentinel.fused)
    fused_metadata = {
        "affine_zyx_um": np.eye(4).tolist(),
        "origin_zyx_um": [0.0, 1.0, 2.0],
        "spacing_zyx_um": [0.4, 0.1, 0.1],
    }
    read_fused_metadata = Mock(return_value=fused_metadata)
    export_ome_tiffs = Mock()
    monkeypatch.setattr(fuseall, "qi2labDataStore", Mock(return_value=datastore))
    monkeypatch.setattr(fuseall, "_load_tile_multichannel_msim", load_tile)
    monkeypatch.setattr(fuseall, "_direct_zarr_fusion_kwargs", direct_options)
    monkeypatch.setattr(fuseall.fusion, "fuse", fuse)
    monkeypatch.setattr(fuseall, "_read_fused_metadata", read_fused_metadata)
    monkeypatch.setattr(fuseall, "export_ome_tiffs", export_ome_tiffs)
    monkeypatch.setattr(fuseall, "tqdm", lambda iterable, **_kwargs: iterable)

    fuseall.fuse_all_channels(
        tmp_path,
        write_ome_tiffs=write_ome_tiffs,
        output_chunk_zyx="32,2048,2048",
        fusion_workers=40,
    )

    channels = ["fiducial", "bit001", "bit002", "bit010"]
    assert load_tile.call_args_list == [
        call(datastore=datastore, tile_id="tile0000", bit_ids=channels[1:]),
        call(datastore=datastore, tile_id="tile0001", bit_ids=channels[1:]),
    ]
    output = fused_root / "full_dataset.ome.zarr"
    fuse.assert_called_once_with(
        images=[sentinel.tile0, sentinel.tile1],
        transform_key="global_registered",
        output_spacing={"z": 0.32, "y": 0.098, "x": 0.098},
        output_chunksize={"z": 32, "y": 2048, "x": 2048},
        output_zarr_url=str(output),
        **fusion_options,
    )
    direct_options.assert_called_once_with(
        misc_utils=fuseall.misc_utils,
        fusion_workers=40,
    )
    read_fused_metadata.assert_called_once_with(sentinel.fused)
    written = datastore._write_extra_attributes.call_args.kwargs
    assert written["image_path"] == output
    assert written["extra_attributes"]["channel_names"] == channels
    assert written["extra_attributes"]["registration_provenance"] == {
        "stage_transform_key": "stage_metadata",
        "chromatic_transform": "chromatic_affine_transforms_zyx_um",
        "local_transform": "local_round_transform_zyx_um",
        "global_transform_key": "global_registered",
        "sofima_applied": False,
    }
    if write_ome_tiffs:
        export_call = export_ome_tiffs.call_args.args
        assert export_call[:4] == (output, fused_root, channels, fused_metadata)
        assert isinstance(export_call[4], fuseall.GlobalRegistrationConfig)
    else:
        export_ome_tiffs.assert_not_called()


def test_fuseall_cli_has_no_gpu_option() -> None:
    assert "gpu_id" not in inspect.signature(fuseall.fuse_all_channels).parameters

    result = CliRunner().invoke(
        fuseall.app,
        ["--help"],
        env={"COLUMNS": "160"},
        terminal_width=160,
    )

    assert result.exit_code == 0
    assert "--gpu-id" not in result.stdout
    assert "--output-chunk-zyx" in result.stdout
    assert "--fusion-workers" in result.stdout
