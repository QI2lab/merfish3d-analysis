"""Normalization sampling with generated signal and background-only tiles."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import merfish3danalysis.PixelDecoder as decoder_module
from merfish3danalysis.PixelDecoder import PixelDecoder


@pytest.fixture
def decoder(monkeypatch):
    result = PixelDecoder.__new__(PixelDecoder)
    result._z_slice = slice(None)
    result._normalization_cell_mask_for_tile = Mock(return_value=None)
    images = {}
    for tile in range(6):
        image = np.full((3, 32, 48), 100, dtype=np.uint16)
        if tile in (2, 4, 5):
            image[1, 12:20, 18:26] = 1000
        images[f"tile{tile}"] = image
    result._datastore = SimpleNamespace(
        tile_ids=[f"tile{i}" for i in range(6)],
        load_local_fiducial_image=Mock(
            side_effect=lambda *, tile, round, return_future: images[tile]
        ),
        load_local_readout_image=Mock(
            side_effect=AssertionError("screen fiducials only")
        ),
    )
    monkeypatch.setattr(decoder_module, "sample", lambda values, count: list(values))
    return result


@pytest.mark.unit
def test_normalization_replaces_empty_tiles_using_only_fiducials(decoder):
    assert decoder._select_normalization_tiles(2) == [2, 4]
    calls = decoder._datastore.load_local_fiducial_image.call_args_list
    pairs = [(c.kwargs["tile"], c.kwargs["round"]) for c in calls]
    assert len(pairs) == len(set(pairs))
    assert pairs == [(f"tile{tile}", 0) for tile in range(5)]
    decoder._datastore.load_local_readout_image.assert_not_called()


@pytest.mark.unit
def test_normalization_exhausts_candidates_without_replacement(decoder):
    with pytest.warns(UserWarning, match="Only 3 of 5"):
        assert decoder._select_normalization_tiles(5) == [2, 4, 5]
    pairs = [
        (c.kwargs["tile"], c.kwargs["round"])
        for c in decoder._datastore.load_local_fiducial_image.call_args_list
    ]
    assert len(pairs) == len(set(pairs))


@pytest.mark.unit
def test_explicit_normalization_candidates_are_restricted_and_deduplicated(decoder):
    assert decoder._select_normalization_tiles(1, [4, 4, 2]) == [4, 2]
    assert {
        c.kwargs["tile"]
        for c in decoder._datastore.load_local_fiducial_image.call_args_list
    } == {"tile4", "tile2"}


@pytest.mark.unit
def test_normalization_rejects_no_signal_and_honors_z_range(decoder):
    decoder._z_slice = slice(0, 1)
    with pytest.raises(ValueError, match="No normalization tiles"):
        decoder._select_normalization_tiles(2)


@pytest.mark.unit
@pytest.mark.parametrize("indices", [[], [-1], [6]])
def test_normalization_rejects_empty_or_invalid_candidate_indices(decoder, indices):
    with pytest.raises(ValueError):
        decoder._select_normalization_tiles(2, indices)
    decoder._datastore.load_local_fiducial_image.assert_not_called()


@pytest.mark.unit
def test_normalization_screens_fiducial_signal_inside_cell_mask(decoder):
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 14:30] = True
    decoder._normalization_cell_mask_for_tile.return_value = mask
    assert decoder._select_normalization_tiles(1, [4]) == [4]
    decoder._normalization_cell_mask_for_tile.return_value = np.zeros_like(mask)
    with pytest.raises(ValueError, match="No normalization tiles"):
        decoder._select_normalization_tiles(1, [4])


@pytest.mark.integration
@pytest.mark.parametrize("deconvolved", [False, True])
def test_normalization_selects_persisted_fiducial_signal(
    tmp_path, monkeypatch, deconvolved
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    path = tmp_path / "qi2labdatastore"
    datastore = qi2labDataStore(path)
    datastore.num_tiles = 3
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.experiment_order = np.array([[1, 1]])
    datastore.voxel_size_zyx_um = [1.5, 0.108, 0.108]
    datastore.codebook = pd.DataFrame([["gene1", 1]], columns=["gene_id", "bit01"])
    datastore.microscope_type = "3D"
    datastore.camera_model = "simulated"
    datastore.tile_overlap = 0.2
    datastore.binning = 1
    datastore.e_per_ADU = 1.0
    datastore.na = 1.35
    datastore.ri = 1.4
    datastore.datastore_state = {**datastore.datastore_state, "Calibrations": True}
    y, x = np.mgrid[:32, :48]
    for tile in range(3):
        image = np.full((3, 32, 48), 100, dtype=np.float32)
        if tile:
            image[1] += 1000 * np.exp(-((y - 16) ** 2 + (x - 24) ** 2) / (2 * 1.5**2))
        datastore.initialize_tile(tile)
        datastore.save_local_corrected_image(
            image.astype(np.uint16), tile=tile, round=0
        )
        if deconvolved:
            datastore.save_local_deconvolved_fiducial_image(
                image.astype(np.uint16), tile=tile, round=0
            )

    decoder = PixelDecoder.__new__(PixelDecoder)
    decoder._datastore = qi2labDataStore(path, validate=False)
    decoder._z_slice = slice(None)
    decoder._normalization_features = "all"
    monkeypatch.setattr(decoder_module, "sample", lambda values, count: list(values))
    assert decoder._select_normalization_tiles(1) == [1]
