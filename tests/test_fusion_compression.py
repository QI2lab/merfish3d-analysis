from pathlib import Path

import numpy as np
import pytest
import zarr
from multiview_stitcher import fusion, misc_utils, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from merfish3danalysis.DataRegistration import _direct_zarr_fusion_kwargs


@pytest.mark.unit
def test_default_fusion_compression_is_explicit_and_lossless() -> None:
    options = _direct_zarr_fusion_kwargs(misc_utils=misc_utils, fusion_workers=1)
    codecs = options["zarr_options"]["zarr_array_creation_kwargs"]["codecs"]

    assert codecs[0].to_dict()["name"] == "bytes"
    compressor = codecs[1].to_dict()
    assert compressor["name"] == "blosc"
    assert compressor["configuration"]["cname"] == "zstd"
    assert compressor["configuration"]["clevel"] == 1
    assert compressor["configuration"]["shuffle"] == "bitshuffle"


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
def test_fusion_rejects_disabled_or_invalid_compression(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _direct_zarr_fusion_kwargs(misc_utils=misc_utils, **kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("fusion_workers", [1, 2])
def test_direct_fusion_preserves_pixels_in_partial_final_z_chunk(
    tmp_path: Path, fusion_workers: int
) -> None:
    # Nineteen planes produce two full seven-plane chunks and a five-plane edge.
    source = np.random.default_rng(7).integers(
        0, 65536, size=(19, 512, 512), dtype=np.uint16
    )
    # Zero background at the acquisition edge avoids undefined zero-weight corners.
    source[:, :4, :] = source[:, -4:, :] = 0
    source[:, :, :4] = source[:, :, -4:] = 0
    array = zarr.create_array(
        tmp_path / "source.zarr", data=source, chunks=(7, 512, 512), zarr_format=3
    )
    sim = si_utils.get_sim_from_array(
        array, dims=("z", "y", "x"), transform_key="global_registered"
    )
    image = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    output = tmp_path / "fused.ome.zarr"
    fusion.fuse(
        images=[image],
        transform_key="global_registered",
        output_zarr_url=str(output),
        **_direct_zarr_fusion_kwargs(
            misc_utils=misc_utils, fusion_workers=fusion_workers
        ),
    )

    reopened = zarr.open_array(output / "0", mode="r")
    assert reopened.shape == (1, 1, 19, 512, 512)
    assert reopened.chunks == (1, 1, 7, 512, 512)
    assert reopened.shards is None
    np.testing.assert_array_equal(reopened[:], source[None, None])


@pytest.mark.integration
@pytest.mark.parametrize("compression", ["blosc-zstd", "blosc-lz4", "zstd"])
def test_direct_fusion_compresses_every_level_without_changing_pixels(
    tmp_path: Path, compression: str
) -> None:
    # Two overlapping acquisitions sample one known scene. Their shared pixels
    # are identical, so normalized overlap blending must reproduce the scene.
    source = np.random.default_rng(2).integers(
        0, 65536, size=(1, 2, 3, 208, 412), dtype=np.uint16
    )
    source[..., :8, :] = source[..., -8:, :] = 0
    source[..., :, :8] = source[..., :, -8:] = 0
    source[0, 0, 1, 10, 10] = 0
    source[0, 1, 1, 10, 10] = 65535
    images = []
    for index in range(2):
        array = zarr.create_array(
            tmp_path / f"tile{index}.zarr",
            data=source[..., 200 * index : 200 * index + 212],
            chunks=(1, 1, 3, 104, 106),
            zarr_format=3,
        )
        sim = si_utils.get_sim_from_array(
            array,
            dims=("t", "c", "z", "y", "x"),
            scale={"z": 1.0, "y": 1.0, "x": 1.0},
            translation={"z": 0.0, "y": 0.0, "x": float(200 * index)},
            affine=np.eye(4),
            transform_key="global_registered",
            c_coords=["fiducial", "bit001"],
        )
        images.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))

    output = tmp_path / "full_dataset.ome.zarr"
    options = _direct_zarr_fusion_kwargs(
        misc_utils=misc_utils,
        fusion_workers=2,
        compression=compression,
        compression_level=2,
    )
    fused = fusion.fuse(
        images=images,
        transform_key="global_registered",
        output_chunksize={"z": 3, "y": 104, "x": 106},
        output_zarr_url=str(output),
        **options,
    )
    group = zarr.open_group(output, mode="r")
    assert group.attrs["ome"]["version"] == "0.5"
    levels = group.attrs["ome"]["multiscales"][0]["datasets"]
    assert list(msi_utils.get_sim_from_msim(fused).c.values) == [
        "fiducial",
        "bit001",
    ]

    assert len(levels) == 3
    expected = source.copy()
    for index, (level, factors) in enumerate(
        zip(levels, [(1, 1), (2, 2), (1, 2)], strict=True)
    ):
        actual = zarr.open_array(output / level["path"], mode="r")
        if index:
            # 208x412 -> 104x206 -> 104x103. Y stops at 104 pixels; the
            # final level averages X pairs. Integer storage truncates fractions.
            dy, dx = factors
            height, width = expected.shape[-2] // dy, expected.shape[-1] // dx
            blocks = expected.reshape(1, 2, 3, height, dy, width, dx)
            expected = blocks.astype(np.float64).sum(axis=(4, 6)) / (dy * dx)
            expected = expected.astype(np.uint16)
        pixels = actual[:]
        x_factor = [1, 2, 4][index]
        overlap = slice(200 // x_factor, 212 // x_factor)
        np.testing.assert_array_equal(
            pixels[..., : overlap.start], expected[..., : overlap.start]
        )
        np.testing.assert_array_equal(
            pixels[..., overlap.stop :], expected[..., overlap.stop :]
        )
        # Floating-point weight normalization followed by uint16 truncation can
        # lower an overlap pixel by one count. Block means preserve this bound.
        error = pixels[..., overlap].astype(np.int64) - expected[..., overlap].astype(
            np.int64
        )
        assert np.all((error >= -1) & (error <= 0))
        assert actual.dtype == np.dtype("uint16")
        scale = next(
            t["scale"]
            for t in level["coordinateTransformations"]
            if t["type"] == "scale"
        )
        np.testing.assert_allclose(scale, [1, 1, 1, [1, 2, 2][index], [1, 2, 4][index]])
        compressor = actual.metadata.to_dict()["codecs"][-1]
        if compression == "zstd":
            assert compressor["name"] == "zstd"
            assert compressor["configuration"]["level"] == 2
        else:
            assert compressor["name"] == "blosc"
            assert compressor["configuration"]["cname"] == compression[6:]
            assert compressor["configuration"]["clevel"] == 2
            assert compressor["configuration"]["shuffle"] == "bitshuffle"
            assert compressor["configuration"]["typesize"] == 2
