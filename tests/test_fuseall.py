import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import fuseall


@pytest.mark.integration
@pytest.mark.parametrize("sharded", [False, True], ids=["unsharded", "sharded"])
def test_fuseall_writes_tiffs_in_fused_directory_after_fiducial_fusion(
    image_datastore, sharded
):
    from yaozarrs import open_group, v06

    from merfish3danalysis.DataRegistration import (
        DataRegistration,
        GlobalRegistrationConfig,
    )
    from merfish3danalysis.viewer.fused import load_fused_channels

    datastore = image_datastore
    storage = {"chunks": (3, 5, 7), "shards": (6, 15, 21)} if sharded else {}
    for tile in range(2):
        stage = [2, 4, tile * 3.2]
        camera = np.diag([1, -1, 1, 1])
        global_affine = np.eye(4)
        global_affine[2, 3] = 0.4
        datastore.save_local_stage_position_zyx_um(stage, camera, tile=tile, round=0)
        datastore.save_global_coord_xforms_um(
            global_affine, stage, datastore.voxel_size_zyx_um, tile=tile
        )
        for round_index in range(2):
            local = np.eye(4)
            if round_index == 1:
                local[2, 3] = 0.4
            datastore.save_local_round_transform_zyx_um(
                local, tile=tile, round=round_index
            )
        image = np.zeros((4, 12, 16), dtype=np.uint16)
        image[2, 5, 5] = 30000 if tile == 0 else 15000
        datastore.save_local_corrected_image(image, tile=tile, round=0, **storage)
        datastore.save_local_deconvolved_fiducial_image(
            image // 2, tile=tile, round=0, **storage
        )
        for bit in range(4):
            readout = np.roll(image, 1, axis=2) if bit in (1, 3) else image
            bit_storage = storage if bit % 2 == 0 else {}
            datastore.save_local_corrected_image(
                readout, tile=tile, bit=bit, **bit_storage
            )
            if bit == 0:
                datastore.save_local_deconvolved_readout_image(
                    readout * 2, tile=tile, bit=bit
                )
    if sharded:
        msim = fuseall._load_tile_multichannel_msim(
            datastore=datastore,
            tile_id=datastore.tile_ids[0],
            bit_ids=datastore.bit_ids,
            spacing_zyx_um=dict(zip("zyx", datastore.voxel_size_zyx_um, strict=True)),
        )
        sim = fuseall.msi_utils.get_sim_from_msim(msim)
        assert fuseall.si_utils.is_xarray_zarr_backed(sim)
        roi = (slice(1, 4), slice(3, 11), slice(5, 16))
        expected = [
            datastore.load_local_deconvolved_fiducial_image(0, 0, return_future=None),
            datastore.load_local_deconvolved_readout_image(0, 0, return_future=None),
            *[
                datastore.load_local_corrected_image(0, bit=bit, return_future=None)
                for bit in range(1, 4)
            ],
        ]
        for channel, source in enumerate(expected):
            np.testing.assert_array_equal(
                sim.isel(t=0, c=channel, z=roi[0], y=roi[1], x=roi[2]).values,
                source[roi].read().result(),
            )
    registration = DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=GlobalRegistrationConfig(fusion_workers=1),
    )
    registration.fuse_global_registered(create_max_proj_tiff=False)
    assert datastore.fused_image_path().parent == (
        datastore.datastore_path / "fiducial_fusion.ome.zarr"
    )

    fuseall.fuse_all_channels(
        datastore.datastore_path,
        write_ome_tiffs=True,
        output_chunk_zyx="4,12,16",
        fusion_workers=2 if sharded else 1,
        compression="blosc-lz4",
        compression_level=2,
    )
    output = datastore.fused_tiff_path()
    group = open_group(datastore.fused_image_path("full_dataset"))
    image = group.ome_metadata(strict=True)
    assert isinstance(image, v06.Image)
    fused_scene = datastore.datastore_path / "fused.ome.zarr"
    scene = open_group(fused_scene).validate(strict=True).ome_metadata()
    relative = str(datastore.fused_image_path("full_dataset").relative_to(fused_scene))
    assert any(
        t.input.path == relative and t.output.name == "world"
        for t in scene.scene.coordinateTransformations
    )
    assert group.attrs["channel_names"] == ["fiducial", *datastore.bit_ids]
    assert group.attrs["registration_provenance"]["sofima_applied"] is False
    for dataset in image.multiscales[0].datasets:
        metadata = group[dataset.path].metadata.model_dump(mode="json")
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"][-3:] == [
            4,
            12,
            16,
        ]
        codec = next(codec for codec in metadata["codecs"] if codec["name"] == "blosc")
        assert codec["configuration"]["cname"] == "lz4"
        assert codec["configuration"]["clevel"] == 2
        assert codec["configuration"]["shuffle"] == "bitshuffle"
    fused, _ = datastore.load_global_fused_image("all_channels", return_future=None)
    peaks = [
        np.unravel_index(np.argmax(fused[0, channel].read().result()), fused.shape[-3:])
        for channel in range(5)
    ]
    assert all(peak == peaks[0] for peak in peaks)
    # The fiducial and first readout use their deconvolved products, while the
    # other readouts fall back to corrected products.
    maxima = [int(fused[0, channel].read().result().max()) for channel in range(5)]
    assert maxima[1] == pytest.approx(4 * maxima[0], abs=4)
    assert maxima[2] == pytest.approx(2 * maxima[0], abs=4)
    stacks = load_fused_channels(datastore.datastore_path)
    full = next(
        stack for stack in stacks if stack.coords["c"][0] == "full_dataset: fiducial"
    )
    assert full.coords["c"] == [
        f"full_dataset: {name}" for name in ["fiducial", *datastore.bit_ids]
    ]
    np.testing.assert_array_equal(full[:], fused.read().result()[0])
    for channel, name in enumerate(["fiducial", *datastore.bit_ids]):
        np.testing.assert_array_equal(
            tifffile.imread(output / f"{name}.ome.tiff"),
            fused[0, channel].read().result().squeeze(),
        )


@pytest.mark.unit
def test_channel_global_transforms_follow_repository_conventions(
    image_datastore,
) -> None:
    datastore = image_datastore
    tile_position = np.asarray((2.0, 10.0, 20.0), dtype=np.float32)
    stage_camera = np.diag([1, -1, 1, 1]).astype(np.float32)
    refinement = np.eye(4, dtype=np.float32)
    refinement[2, 3] = 4.0
    moving = np.eye(4, dtype=np.float32)
    moving[1, 3] = 1.5
    datastore.save_local_round_transform_zyx_um(np.eye(4), tile=0, round=0)
    datastore.save_local_round_transform_zyx_um(moving, tile=0, round=1)
    chromatic = np.eye(4, dtype=np.float32)
    chromatic[2, 3] = 0.25
    datastore.save_chromatic_affine_transforms_zyx_um(
        {
            "estimator": "decoded_rna_on_bit_weighted_centroid_yx_affine_graph",
            "decode_run_key": "previous",
            "channels": {"red": {"wavelength_um": 0.67, "affine_zyx_um": chromatic}},
        }
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
    expected = np.stack(
        [
            refinement @ stage_camera,
            refinement @ stage_camera,
            refinement
            @ stage_camera
            @ tile_origin
            @ np.linalg.inv(moving)
            @ np.linalg.inv(tile_origin),
        ]
    )
    np.testing.assert_allclose(transforms, expected, atol=1e-6)


@pytest.mark.unit
def test_fuseall_rejects_gpu_option_before_loading_data(image_datastore):
    result = CliRunner().invoke(
        fuseall.app, [str(image_datastore.datastore_path), "--gpu-id", "0"]
    )
    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        ["--compression", "none"],
        ["--compression", "unknown"],
        ["--compression-level", "0"],
        ["--compression-level", "10"],
    ],
)
def test_fuseall_rejects_invalid_compression_before_loading_data(
    image_datastore, options: list[str]
) -> None:
    result = CliRunner().invoke(
        fuseall.app, [str(image_datastore.datastore_path), *options]
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output
