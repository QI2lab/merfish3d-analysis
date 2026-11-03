"""Fusion compression and pyramid behavior through the datastore API."""

import numpy as np
import pytest
from multiview_stitcher import misc_utils
from yaozarrs import open_group, v06

from merfish3danalysis.cli.qi2lab_microscopes.fuseall import fuse_all_channels
from merfish3danalysis.DataRegistration import _direct_zarr_fusion_kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"compression": "none"},
        {"compression": "unknown"},
        {"compression_level": 0},
        {"compression_level": 10},
    ],
)
def test_fusion_rejects_disabled_or_invalid_compression(kwargs):
    with pytest.raises(ValueError):
        _direct_zarr_fusion_kwargs(misc_utils=misc_utils, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("fusion_type", ["fiducial", "all_channels"])
def test_fused_product_rejects_stale_publication(image_datastore, fusion_type):
    from yaozarrs.write.v06 import write_image

    from merfish3danalysis.storage.ome import ConcurrentSceneUpdateError

    datastore = image_datastore
    pixels = np.arange(48, dtype=np.uint16).reshape(3, 4, 4)
    kwargs = {
        "affine_zyx_um": np.eye(4),
        "origin_zyx_um": [0, 0, 0],
        "spacing_zyx_um": datastore.voxel_size_zyx_um,
        "fusion_type": fusion_type,
    }
    datastore.save_global_fiducial_image(pixels, **kwargs)
    scene_path = datastore.fused_scene_path(fusion_type)
    image_path = datastore.fused_image_path(
        None if fusion_type == "fiducial" else "full_dataset"
    )
    image = open_group(image_path).ome_metadata(strict=True)
    staged_path, revision, expected = datastore.prepare_global_fused_image(fusion_type)
    write_image(staged_path, image, pixels, writer="tensorstore")

    datastore.save_global_fiducial_image(pixels + 1, **kwargs)
    with pytest.raises(ConcurrentSceneUpdateError):
        datastore.publish_global_fused_image(fusion_type, image, revision, expected)

    assert staged_path.parent == scene_path
    actual, _ = datastore.load_global_fused_image(fusion_type, return_future=False)
    np.testing.assert_array_equal(actual, pixels + 1)


@pytest.mark.integration
@pytest.mark.parametrize(
    "compression, level, workers",
    [("blosc-zstd", 1, 1), ("blosc-lz4", 2, 1), ("zstd", 7, 2)],
)
def test_multichannel_fusion_pyramid_preserves_pixels_and_codec(
    image_datastore, compression, level, workers
):
    datastore = image_datastore
    pixels = np.random.default_rng(7).integers(0, 20000, (3, 20, 210), dtype=np.uint16)
    pixels[:, :2, :] = pixels[:, -2:, :] = 0
    pixels[:, :, :2] = pixels[:, :, -2:] = 0
    for tile in range(2):
        datastore.save_local_stage_position_zyx_um(
            [0, 0, 0], np.eye(4), tile=tile, round=0
        )
        datastore.save_global_coord_xforms_um(
            np.eye(4), [0, 0, 0], datastore.voxel_size_zyx_um, tile=tile
        )
        for round_index in range(2):
            datastore.save_local_round_transform_zyx_um(
                np.eye(4), tile=tile, round=round_index
            )
        datastore.save_local_corrected_image(pixels, tile=tile, round=0)
        for bit in range(4):
            datastore.save_local_corrected_image(pixels + bit, tile=tile, bit=bit)

    fuse_all_channels(
        datastore.datastore_path,
        output_chunk_zyx="3,16,64",
        fusion_workers=workers,
        compression=compression,
        compression_level=level,
    )

    fused_scene = datastore.fused_scene_path("all_channels")
    assert fused_scene == datastore.datastore_path / "fused.ome.zarr"
    assert isinstance(
        open_group(fused_scene).validate(strict=True).ome_metadata(), v06.Scene
    )
    assert not (datastore.datastore_path / "fiducial_fusion.ome.zarr").exists()
    assert datastore.fused_image_path("full_dataset").is_relative_to(fused_scene)
    group = open_group(datastore.fused_image_path("full_dataset")).validate(strict=True)
    model = group.ome_metadata(strict=True)
    assert isinstance(model, v06.Image)
    datasets = model.multiscales[0].datasets
    assert len(datasets) == 2
    full = group[datasets[0].path].to_tensorstore().read().result()
    reduced = group[datasets[1].path].to_tensorstore().read().result()
    np.testing.assert_allclose(full[0, 0, :, 2:-2, 2:-2], pixels[:, 2:-2, 2:-2], atol=1)
    trimmed = full[..., : reduced.shape[-1] * 2]
    expected = trimmed.reshape(*trimmed.shape[:-1], reduced.shape[-1], 2).mean(axis=-1)
    np.testing.assert_array_equal(reduced, expected.astype(full.dtype))
    for dataset in datasets:
        metadata = group[dataset.path].metadata.model_dump(mode="json")
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [
            1,
            1,
            3,
            16,
            64,
        ]
        compressor = metadata["codecs"][-1]
        if compression == "zstd":
            assert compressor["name"] == "zstd"
            assert compressor["configuration"]["level"] == level
            assert compressor["configuration"]["checksum"] is False
        else:
            assert compressor["name"] == "blosc"
            assert compressor["configuration"]["cname"] == compression.removeprefix(
                "blosc-"
            )
            assert compressor["configuration"]["clevel"] == level
            assert compressor["configuration"]["shuffle"] == "bitshuffle"
            assert compressor["configuration"]["typesize"] == full.dtype.itemsize
            assert compressor["configuration"]["blocksize"] == 0

    if workers > 1:
        fuse_all_channels(
            datastore.datastore_path,
            output_chunk_zyx="3,16,64",
            fusion_workers=1,
            compression=compression,
            compression_level=level,
        )
        serial = open_group(datastore.fused_image_path("full_dataset"))
        for dataset, parallel_pixels in zip(datasets, (full, reduced), strict=True):
            serial_pixels = serial[dataset.path].to_tensorstore().read().result()
            np.testing.assert_array_equal(parallel_pixels, serial_pixels)
