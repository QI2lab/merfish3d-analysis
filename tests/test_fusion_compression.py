from pathlib import Path

import numpy as np
import pytest
import zarr
from multiview_stitcher import fusion, misc_utils, msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from merfish3danalysis.DataRegistration import _direct_zarr_fusion_kwargs


def test_default_fusion_compression_is_explicit_and_lossless() -> None:
    options = _direct_zarr_fusion_kwargs(misc_utils=misc_utils, fusion_workers=1)
    codecs = options["zarr_options"]["zarr_array_creation_kwargs"]["codecs"]

    assert codecs[0].to_dict()["name"] == "bytes"
    compressor = codecs[1].to_dict()
    assert compressor["name"] == "blosc"
    assert compressor["configuration"]["cname"] == "zstd"
    assert compressor["configuration"]["clevel"] == 1
    assert compressor["configuration"]["shuffle"] == "bitshuffle"


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


@pytest.mark.parametrize("compression", ["blosc-zstd", "blosc-lz4", "zstd"])
def test_direct_fusion_compresses_every_level_without_changing_pixels(
    tmp_path: Path, compression: str
) -> None:
    # Real Zarr-backed MVS inputs, overlapping tiles and two channels. The
    # lateral extent is large enough to exercise actual pyramid creation.
    source = np.random.default_rng(2).integers(
        0, 65536, size=(1, 2, 3, 208, 212), dtype=np.uint16
    )
    source[0, 0, 1, 10, 10] = 0
    source[0, 1, 1, 10, 10] = 65535
    images = []
    for index in range(2):
        array = zarr.create_array(
            tmp_path / f"tile{index}.zarr",
            data=source,
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
        fusion_workers=1,
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
    assert len(levels) >= 2
    assert list(msi_utils.get_sim_from_msim(fused).c.values) == [
        "fiducial",
        "bit001",
    ]

    # Compare against the previous plain-Zstd output, including interpolation,
    # overlap blending, uint16 extremes, channel order and downsampling.
    baseline = tmp_path / "baseline.ome.zarr"
    baseline_options = _direct_zarr_fusion_kwargs(
        misc_utils=misc_utils,
        fusion_workers=1,
        compression="zstd",
        compression_level=3,
    )
    fusion.fuse(
        images=images,
        transform_key="global_registered",
        output_chunksize={"z": 3, "y": 104, "x": 106},
        output_zarr_url=str(baseline),
        **baseline_options,
    )
    for level in levels:
        actual = zarr.open_array(output / level["path"], mode="r")
        expected = zarr.open_array(baseline / level["path"], mode="r")
        np.testing.assert_array_equal(actual[:], expected[:])
        assert actual.dtype == np.dtype("uint16")
        assert actual.chunks == expected.chunks
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
