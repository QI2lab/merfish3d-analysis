import json
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from ndstorage import NDTiffDataset

from merfish3danalysis.cli.qi2lab_microscopes import create_datastore
from merfish3danalysis.cli.qi2lab_microscopes.create_datastore import (
    _readout_bit_ids,
    _sample_readout_tile_bit_pairs,
)
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.unit
def test_readout_bit_ids_groups_bits_by_acquisition_channel() -> None:
    experiment_order = np.asarray(
        [
            [1, 1, 2],
            [2, 3, 4],
            [3, 5, 6],
        ],
        dtype=np.int64,
    )
    bit_ids = [f"bit{bit_number:03d}" for bit_number in range(1, 7)]

    assert _readout_bit_ids(experiment_order, 1, bit_ids) == [
        "bit001",
        "bit003",
        "bit005",
    ]
    assert _readout_bit_ids(experiment_order, 2, bit_ids) == [
        "bit002",
        "bit004",
        "bit006",
    ]


@pytest.mark.unit
def test_readout_bit_ids_keeps_each_bit_once() -> None:
    experiment_order = np.asarray([[1, 1], [2, 2], [3, 1]], dtype=np.int64)

    assert _readout_bit_ids(experiment_order, 1, ["bit001", "bit002"]) == [
        "bit001",
        "bit002",
    ]


@pytest.mark.unit
def test_sample_readout_tile_bit_pairs_uses_unique_tiles_across_bits() -> None:
    pairs = _sample_readout_tile_bit_pairs(
        ["bit001", "bit003", "bit005"],
        num_tiles=4,
        max_images=4,
        rng=np.random.default_rng(4),
    )

    tile_indices = [tile_idx for tile_idx, _bit_id in pairs]
    sampled_bit_ids = [bit_id for _tile_idx, bit_id in pairs]
    assert len(pairs) == 4
    assert set(tile_indices) == {0, 1, 2, 3}
    assert len(tile_indices) == len(set(tile_indices))
    assert set(sampled_bit_ids) == {"bit001", "bit003", "bit005"}


@pytest.mark.unit
@pytest.mark.parametrize("num_tiles,limit", [(2, 10), (448, 100)])
def test_sample_readout_tile_bit_pairs_caps_samples_at_the_number_of_tiles(
    num_tiles, limit
) -> None:
    pairs = _sample_readout_tile_bit_pairs(
        ["bit001", "bit003"],
        num_tiles=num_tiles,
        max_images=limit,
        rng=np.random.default_rng(2),
    )

    assert len(pairs) == min(num_tiles, limit)
    assert len({tile_idx for tile_idx, _bit_id in pairs}) == len(pairs)
    assert {bit_id for _tile_idx, bit_id in pairs} <= {"bit001", "bit003"}
    if num_tiles > limit:
        assert max(tile_idx for tile_idx, _bit_id in pairs) >= limit


@pytest.mark.unit
def test_sample_readout_tile_bit_pairs_requires_a_positive_limit() -> None:
    with pytest.raises(ValueError, match="max_images"):
        _sample_readout_tile_bit_pairs(
            ["bit001"],
            num_tiles=1,
            max_images=0,
            rng=np.random.default_rng(1),
        )


@pytest.mark.unit
@pytest.mark.parametrize("limit,empty_count", [(1, 0), (2, 0), (2, 1), (2, 2)])
def test_fiducial_sampling_is_capped_but_corrects_every_tile(
    tmp_path, monkeypatch, limit, empty_count
):
    metadata = {
        "Camera-CameraName": "C13440-20CU",
        "Camera-CONVERSION FACTOR COEFF": 1.0,
        "Camera-CONVERSION FACTOR OFFSET": 0.0,
        "PixelSizeUm": 0.2,
        "PixelSizeAffine": "0.2;0;0;0;0.2;0",
        "YPosition_um_Intended": 0.0,
        "XYStage-TransposeMirrorX": "0",
        "XYStage-TransposeMirrorY": "0",
    }
    channels = ("F-Blue", "F-Yellow", "F-Red")
    for tile in range(2):
        base = f"acquisition_r0001_tile{tile:04d}"
        raw_path = tmp_path / f"{base}_1"
        acquisition = NDTiffDataset(str(raw_path), summary_metadata={}, writable=True)
        for z in range(3):
            for channel in channels:
                pixels = np.full((32, 48), 100 + tile, dtype=np.uint16)
                if channel == "F-Blue" and tile >= empty_count:
                    pixels[12:20, 18:26] += 1000
                acquisition.put_image(
                    {"z": z, "channel": channel},
                    pixels,
                    {
                        **metadata,
                        "XPosition_um_Intended": tile * 8.0,
                        "ZPosition_um_Intended": float(z),
                    },
                )
        acquisition.finish()
        (raw_path / f"{base}_NDTiffStack.tif").symlink_to("NDTiffStack.tif")
    (tmp_path / "scan_metadata.csv").write_text(
        "root_name,num_r,num_xyz,num_ch,blue_active,yellow_active,red_active,"
        "channels_reversed,z_step_um,yx_pixel_um,binning\n"
        "acquisition,1,2,3,True,True,True,False,1.0,0.2,1\n"
    )
    pd.DataFrame({"round": [1], "bit01": [1], "bit02": [2]}).to_csv(
        tmp_path / "bit_order.csv", index=False
    )
    pd.DataFrame({"gene_id": ["gene"], "bit01": [1], "bit02": [1]}).to_csv(
        tmp_path / "codebook.csv", index=False
    )

    samples = []

    def estimate(images):
        samples.append([int(image.result()[0, 0, 0]) - 100 for image in images])
        return np.full((32, 48), 0.5, dtype=np.float32)

    monkeypatch.setattr(create_datastore, "estimate_shading", estimate)
    available = 2 - empty_count
    if not available:
        context = pytest.raises(ValueError, match="No fiducial tiles")
    elif available < min(2, limit):
        context = pytest.warns(UserWarning, match="Only")
    else:
        context = nullcontext()
    with context:
        create_datastore.convert_data(
            tmp_path,
            save_illuminations=False,
            max_flatfield_images=limit,
            noise_map_shape_yx=(32, 48),
        )

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    assert json.loads(
        (datastore.datastore_path / "flatfield_tile_ids.json").read_text()
    ) == [datastore.tile_ids[tile] for tile in range(empty_count, 2)]
    if not available:
        assert samples == []
        return
    assert len(samples) == len(channels)
    for sampled in samples:
        assert len(sampled) == len(set(sampled)) == min(available, limit)
        assert set(sampled) <= set(range(empty_count, 2))
    for tile in range(2):
        for selection in ({"round": 0}, {"bit": 0}, {"bit": 1}):
            actual = datastore.load_local_corrected_image(
                tile=tile, return_future=False, **selection
            )
            expected = np.full((3, 32, 48), 2 * (100 + tile), dtype=np.uint16)
            if "round" in selection and tile >= empty_count:
                expected[:, 12:20, 18:26] += 2000
            np.testing.assert_array_equal(actual, expected)
            assert datastore.load_local_image_metadata(tile=tile, **selection)[
                "shading_correction"
            ]
