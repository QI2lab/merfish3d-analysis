import json
import weakref
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from merfish3danalysis.cli.qi2lab_microscopes import create_datastore
from merfish3danalysis.cli.qi2lab_microscopes.create_datastore import (
    _readout_bit_ids,
    _sample_readout_tile_bit_pairs,
)


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
def test_sample_readout_tile_bit_pairs_caps_samples_at_the_number_of_tiles() -> None:
    pairs = _sample_readout_tile_bit_pairs(
        ["bit001", "bit003"],
        num_tiles=2,
        max_images=10,
        rng=np.random.default_rng(2),
    )

    assert len(pairs) == 2
    assert {tile_idx for tile_idx, _bit_id in pairs} == {0, 1}
    assert {bit_id for _tile_idx, bit_id in pairs} == {"bit001", "bit003"}


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
@pytest.mark.parametrize(
    "num_tiles,limit,empty_count",
    [
        (2, 1, 0),
        (2, 100, 0),
        (100, 100, 0),
        (448, 100, 0),
        (6, 2, 3),
        (6, 5, 4),
        (6, 2, 6),
    ],
)
@pytest.mark.parametrize("num_ch", [1, 3])
def test_fiducial_sampling_is_capped_but_corrects_every_tile(
    monkeypatch, num_tiles, limit, empty_count, num_ch
):
    root = Path("/mock/acquisition")
    monkeypatch.setattr(Path, "exists", Mock(return_value=True))
    save_ids = Mock()
    monkeypatch.setattr(Path, "write_text", save_ids)
    monkeypatch.setattr(pd, "read_csv", Mock(return_value=pd.DataFrame([[1, 1, 2]])))
    monkeypatch.setattr(
        create_datastore,
        "read_metadatafile",
        Mock(
            return_value={
                "root_name": "test",
                "num_r": 1,
                "num_xyz": num_tiles,
                "num_ch": num_ch,
                "blue_active": True,
                "yellow_active": num_ch == 3,
                "red_active": num_ch == 3,
                "channels_reversed": False,
                "z_step_um": 1.5,
                "yx_pixel_um": 0.1,
                "binning": 1,
            }
        ),
    )
    dataset = Mock()
    dataset.get_image_coordinates_list.return_value = [{"channel": "F-Blue"}]
    dataset.read_metadata.return_value = {
        "Camera-CameraName": "C13440-20CU",
        "Camera-CONVERSION FACTOR COEFF": 0.5,
        "Camera-CONVERSION FACTOR OFFSET": 100,
        "PixelSizeUm": 0.1,
        "PixelSizeAffine": "0.1;0;0;0;0.1;0",
    }
    monkeypatch.setattr(
        create_datastore, "_load_dataset_silently", Mock(return_value=dataset)
    )
    monkeypatch.setattr(
        create_datastore,
        "imread",
        Mock(return_value=np.ones((1, 1, 4, 6), dtype=np.uint16)),
    )
    monkeypatch.setattr(
        create_datastore, "generate_qi2lab_psf", Mock(return_value=np.ones((1, 3, 3)))
    )
    datastore = Mock(
        num_tiles=num_tiles,
        num_rounds=1,
        bit_ids=["bit001", "bit002"] if num_ch == 3 else [],
        tile_ids=[f"tile{i:04d}" for i in range(num_tiles)],
    )
    candidates = list(range(num_tiles))
    empty_tiles = set(candidates[:empty_count])
    checked = []

    readout_tiles = []
    live_images = []
    samples = []

    def load_image(*, tile, round=None, bit=None, return_future=True):
        if not samples or return_future:
            assert all(ref() is None for ref in live_images)
        assert isinstance(tile, int)
        assert (round == 0 and bit is None) or bit in datastore.bit_ids
        image = np.full((1, 32, 48), 100 + tile, dtype=np.uint16)
        if round == 0 and tile not in empty_tiles:
            image[:, 12:20, 18:26] += 1000
        if not samples or return_future:
            live_images.append(weakref.ref(image))
        if not return_future:
            if not samples:
                checked.append(tile)
            return image
        future = Future()
        future.set_result(image)
        if bit is not None:
            readout_tiles.append(tile)
        return future

    datastore.load_local_corrected_image.side_effect = load_image
    monkeypatch.setattr(
        create_datastore, "qi2labDataStore", Mock(return_value=datastore)
    )

    def estimate(images):
        assert all(ref() is None for ref in live_images)
        samples.append([])
        for image in images:
            samples[-1].append(int(image.result()[0, 0, 0]) - 100)
            del image
        assert all(ref() is None for ref in live_images)
        return np.full((32, 48), 0.5, dtype=np.float32)

    estimator = Mock(side_effect=estimate)
    monkeypatch.setattr(create_datastore, "estimate_shading", estimator)

    available = num_tiles - empty_count
    expected_count = min(available, limit)
    if not available:
        context = pytest.raises(ValueError, match="No fiducial tiles")
    elif available < min(num_tiles, limit):
        context = pytest.warns(UserWarning, match="Only")
    else:
        context = nullcontext()
    with context:
        create_datastore.convert_data(
            root,
            output_path=root,
            save_illuminations=False,
            max_flatfield_images=limit,
            noise_map_shape_yx=(4, 6),
        )
    assert len(checked) == len(set(checked))
    assert checked == candidates
    assert json.loads(save_ids.call_args.args[0]) == [
        datastore.tile_ids[tile] for tile in candidates if tile not in empty_tiles
    ]
    if not available:
        estimator.assert_not_called()
        datastore.save_local_corrected_image.assert_not_called()
        assert len(checked) == num_tiles
        return

    assert estimator.call_count == num_ch
    tiles = samples[0]
    assert len(tiles) == expected_count
    assert not set(tiles) & empty_tiles
    assert len(set(tiles)) == len(tiles)
    assert all(0 <= tile < num_tiles for tile in tiles)
    assert not set(readout_tiles) & empty_tiles
    assert len(readout_tiles) == (num_ch - 1) * len(tiles)
    for sampled in samples:
        assert len(sampled) == len(set(sampled)) == expected_count
        assert set(sampled) <= set(candidates) - empty_tiles
        if num_tiles > 100 and not empty_count:
            # Every channel samples the eligible pool, not its first 100 IDs.
            assert max(sampled) >= limit
    writes = datastore.save_local_corrected_image.call_args_list
    assert len(writes) == num_tiles * num_ch
    assert {call.kwargs["tile"] for call in writes} == set(range(num_tiles))
    for call in writes:
        tile = call.kwargs["tile"]
        assert call.kwargs["shading_correction"] is True
        expected = np.full((1, 32, 48), 2 * (100 + tile))
        if call.kwargs.get("round") == 0 and tile not in empty_tiles:
            expected[:, 12:20, 18:26] += 2000
        np.testing.assert_array_equal(call.args[0], expected)
