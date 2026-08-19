import numpy as np
import pytest

from merfish3danalysis.cli.qi2lab_microscopes.create_datastore import (
    _readout_bit_ids,
    _sample_readout_tile_bit_pairs,
)


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


def test_readout_bit_ids_keeps_each_bit_once() -> None:
    experiment_order = np.asarray([[1, 1], [2, 2], [3, 1]], dtype=np.int64)

    assert _readout_bit_ids(experiment_order, 1, ["bit001", "bit002"]) == [
        "bit001",
        "bit002",
    ]


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


def test_sample_readout_tile_bit_pairs_requires_a_positive_limit() -> None:
    with pytest.raises(ValueError, match="max_images"):
        _sample_readout_tile_bit_pairs(
            ["bit001"],
            num_tiles=1,
            max_images=0,
            rng=np.random.default_rng(1),
        )
