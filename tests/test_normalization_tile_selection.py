"""Normalization sampling with stored signal and background-only tiles."""

import numpy as np
import pytest
from roifile import ImagejRoi, roiwrite

import merfish3danalysis.PixelDecoder as decoder_module
from merfish3danalysis.PixelDecoder import PixelDecoder

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.parametrize("image_datastore", ["six_tiles"], indirect=True),
]


@pytest.fixture
def decoder(image_datastore, monkeypatch):
    for tile in range(6):
        image = np.full((3, 32, 48), 100, dtype=np.uint16)
        if tile in (2, 4, 5):
            image[1, 12:20, 18:26] = 1000
        image_datastore.save_local_corrected_image(image, tile=tile, round=0)
    monkeypatch.setattr(decoder_module, "sample", lambda values, count: list(values))
    return PixelDecoder(image_datastore, merfish_bits=4, num_gpus=0, verbose=0)


@pytest.mark.unit
def test_normalization_replaces_empty_tiles_using_only_fiducials(decoder):
    assert decoder._select_normalization_tiles(2) == [2, 4]


@pytest.mark.unit
def test_normalization_exhausts_candidates_without_replacement(decoder):
    with pytest.warns(UserWarning, match="Only 3 of 5"):
        assert decoder._select_normalization_tiles(5) == [2, 4, 5]


@pytest.mark.unit
def test_explicit_normalization_candidates_are_restricted_and_deduplicated(decoder):
    assert decoder._select_normalization_tiles(1, [4, 4, 2]) == [4, 2]


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


@pytest.mark.unit
@pytest.mark.parametrize("has_cell", [False, True])
def test_normalization_screens_fiducial_signal_inside_cell_mask(decoder, has_cell):
    roi_path = (
        decoder._datastore.datastore_path
        / "segmentation/cellpose/imagej_rois/global_coords_rois.zip"
    )
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    rois = (
        [ImagejRoi.frompoints([[6, 2], [12, 2], [12, 5], [6, 5]])] if has_cell else []
    )
    roiwrite(roi_path, rois, mode="w")
    if has_cell:
        assert decoder._select_normalization_tiles(1, [4]) == [4]
    else:
        with pytest.raises(ValueError, match="No normalization tiles"):
            decoder._select_normalization_tiles(1, [4])


@pytest.mark.unit
@pytest.mark.parametrize("deconvolved", [False, True])
def test_normalization_selects_persisted_fiducial_signal(decoder, deconvolved):
    datastore = decoder._datastore
    if deconvolved:
        for tile in range(datastore.num_tiles):
            image = datastore.load_local_corrected_image(
                tile=tile, round=0, return_future=False
            )
            datastore.save_local_deconvolved_fiducial_image(image, tile=tile, round=0)
    assert decoder._select_normalization_tiles(1) == [2]
