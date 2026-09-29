"""Zhuang illumination correction with generated images and real persistence."""

import json
import runpy
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from tifffile import TiffFile
from yaozarrs import open_group

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.fixture
def zhuang_conversion(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psfmodels",
        SimpleNamespace(make_psf=Mock(return_value=np.ones((1, 3, 3)))),
    )
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "examples/zhuang_lab/01_convert_to_qi2lab.py"
        )
    )
    convert = example["convert_data"]
    codebook = pd.DataFrame(
        {"id": [1], "name": ["gene1"], **{f"RS{i}": [1] for i in range(38)}}
    )
    monkeypatch.setattr(
        pd,
        "read_csv",
        Mock(side_effect=[codebook, pd.DataFrame([[0, 1, 2], [3, 4, 5]])]),
    )
    monkeypatch.setattr(
        Path,
        "glob",
        Mock(return_value=[tmp_path / "tile0.tif", tmp_path / "tile1.tif"]),
    )
    # Raw CZYX is transposed on conversion. After offset/gain correction,
    # channels alternate 46/92 electrons on tile 0 and 92/184 on tile 1.
    raw = np.full((39, 2, 3, 4), 200, dtype=np.uint16)
    raw[1:38:2] = 300
    raw[:, :, 0, 0] = 50  # below camera offset
    raw[:, :, 2, 3] = 65535  # exercises saturation after shading correction
    second = raw.copy()
    second[raw == 200] = 300
    second[raw == 300] = 500
    monkeypatch.setitem(convert.__globals__, "imread", Mock(side_effect=[raw, second]))
    # Pixel persistence is isolated from the separately tested signal detector.
    monkeypatch.setitem(
        convert.__globals__, "image_has_signal", Mock(return_value=True)
    )
    return convert


@pytest.mark.integration
@pytest.mark.parametrize("max_images", [1, 2, 100])
def test_zhuang_illumination_corrects_and_persists_each_channel(
    zhuang_conversion, tmp_path, monkeypatch, max_images
):
    flatfields = np.ones((3, 4, 3), dtype=np.float32)
    flatfields[0, :, 1:] = 0.5
    flatfields[1, :, 1:] = 0.25
    flatfields[2, :, 1:] = 0.125
    samples = []

    def estimate(images):
        samples.append([image.result() for image in images])
        return flatfields[len(samples) - 1]

    monkeypatch.setitem(zhuang_conversion.__globals__, "estimate_shading", estimate)
    output = tmp_path / "qi2labdatastore"
    zhuang_conversion(tmp_path, output_path=output, max_flatfield_images=max_images)

    # Estimation is capped for every channel; all tiles are still corrected below.
    assert [len(channel) for channel in samples] == [min(max_images, 2)] * 3
    if max_images == 1:
        # Seeded random selection chooses the second eligible tile, not the first.
        assert int(samples[0][0][0, 1, 0]) == 92
    for channel, values in zip(samples, [(46, 92), (46, 92), (92, 184)], strict=True):
        intensities = [int(image[0, 1, 0]) for image in channel]
        assert len(set(intensities)) == len(intensities)
        assert set(intensities) <= set(values)
    assert all(image.shape == (2, 4, 3) for channel in samples for image in channel)

    datastore = qi2labDataStore(output, validate=False)
    assert json.loads((output / "flatfield_tile_ids.json").read_text()) == list(
        datastore.tile_ids
    )
    assert datastore.datastore_state["Corrected"] is True
    for tile in range(2):
        for kind, count in [("round", 19), ("bit", 38)]:
            for index in range(count):
                channel = 0 if kind == "round" else 1 + index % 2
                # Independently computed from camera gain 0.46 e-/ADU,
                # offset 100 ADU and the channel flatfields above.
                baseline = (92 if channel == 2 else 46) * (tile + 1)
                expected = np.full((2, 4, 3), baseline, dtype=np.uint16)
                expected[:, :, 1:] = baseline * (2, 4, 8)[channel]
                expected[:, 0, 0] = 0
                expected[:, 3, 2] = (60200, 65535, 65535)[channel]
                selection = {kind: index}
                actual = datastore.load_local_corrected_image(
                    tile=tile, return_future=False, **selection
                )
                np.testing.assert_array_equal(actual, expected)
                path = datastore.local_image_path(tile, "corrected_data", **selection)
                metadata = datastore.load_image_metadata(path)
                assert metadata["shading_correction"] is True
                assert metadata["gain_correction"] is True
                assert metadata["hotpixel_correction"] is False
                assert metadata["psf_idx"] == channel
                group = open_group(path)
                assert group.metadata.zarr_format == 3
                multiscale = group.attrs["ome"]["multiscales"][0]
                scale = multiscale["datasets"][0]["coordinateTransformations"][0]
                np.testing.assert_allclose(scale["scale"], [1.5, 0.108, 0.108])

    with TiffFile(tmp_path / "illuminations.ome.tif") as tif:
        np.testing.assert_array_equal(tif.asarray(), flatfields)
        assert tif.series[0].axes == "CYX"
        pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        for axis in "XY":
            assert float(pixels.attrib[f"PhysicalSize{axis}"]) == pytest.approx(0.108)
            assert pixels.attrib[f"PhysicalSize{axis}Unit"] == "µm"


@pytest.mark.integration
def test_zhuang_failed_estimation_does_not_mark_conversion_complete(
    zhuang_conversion, tmp_path, monkeypatch
):
    monkeypatch.setitem(
        zhuang_conversion.__globals__,
        "estimate_shading",
        Mock(
            side_effect=[
                np.ones((4, 3), dtype=np.float32),
                np.ones((4, 3), dtype=np.float32),
                RuntimeError("estimation failed"),
            ]
        ),
    )
    output = tmp_path / "qi2labdatastore"
    with pytest.raises(RuntimeError, match="estimation failed"):
        zhuang_conversion(tmp_path, output_path=output)
    datastore = qi2labDataStore(output, validate=False)
    assert datastore.datastore_state["Corrected"] is False


@pytest.mark.integration
def test_zhuang_reuses_signal_fiducial_tiles_for_all_flatfields(
    zhuang_conversion, tmp_path, monkeypatch
):
    # The first candidate has 46 e- at this pixel, the second 92 e-.
    screen = Mock(side_effect=lambda image: int(image[0, 1, 0]) == 92)
    monkeypatch.setitem(zhuang_conversion.__globals__, "image_has_signal", screen)
    samples = []

    def estimate(images):
        samples.append([image.result() for image in images])
        return np.ones((4, 3), dtype=np.float32)

    monkeypatch.setitem(zhuang_conversion.__globals__, "estimate_shading", estimate)
    output = tmp_path / "qi2labdatastore"
    zhuang_conversion(tmp_path, output_path=output, max_flatfield_images=1)

    assert screen.call_count == 2  # only the two fiducials, never readout images
    assert [len(images) for images in samples] == [1, 1, 1]
    assert [int(images[0][0, 1, 0]) for images in samples] == [92, 92, 184]
    datastore = qi2labDataStore(output, validate=False)
    assert datastore.datastore_state["Corrected"] is True
    assert json.loads((output / "flatfield_tile_ids.json").read_text()) == [
        datastore.tile_ids[1]
    ]
    for tile in range(2):
        for kind in ("round", "bit"):
            path = datastore.local_image_path(tile, "corrected_data", **{kind: 0})
            assert datastore.load_image_metadata(path)["shading_correction"] is True


@pytest.mark.integration
def test_zhuang_no_signal_stops_before_flatfield_fitting(
    zhuang_conversion, tmp_path, monkeypatch
):
    screen = Mock(return_value=False)
    estimate = Mock()
    monkeypatch.setitem(zhuang_conversion.__globals__, "image_has_signal", screen)
    monkeypatch.setitem(zhuang_conversion.__globals__, "estimate_shading", estimate)
    output = tmp_path / "qi2labdatastore"
    with pytest.raises(ValueError, match="No fiducial tiles"):
        zhuang_conversion(tmp_path, output_path=output, max_flatfield_images=1)
    assert screen.call_count == 2
    estimate.assert_not_called()
    assert json.loads((output / "flatfield_tile_ids.json").read_text()) == []
    assert qi2labDataStore(output, validate=False).datastore_state["Corrected"] is False


@pytest.mark.integration
def test_zhuang_releases_screened_and_sampled_stacks(
    zhuang_conversion, tmp_path, monkeypatch
):
    load = qi2labDataStore.load_local_corrected_image
    live_images = []
    fits = []

    def load_image(datastore, *args, **kwargs):
        assert all(ref() is None for ref in live_images)
        result = load(datastore, *args, **kwargs)
        if not fits:
            live_images.append(weakref.ref(result))
        return result

    def estimate(images):
        assert all(ref() is None for ref in live_images)
        fits.append([])
        for future in images:
            image = future.result()
            live_images.append(weakref.ref(image))
            fits[-1].append(image.max(axis=0))
            del image, future
        assert all(ref() is None for ref in live_images)
        return np.ones((4, 3), dtype=np.float32)

    monkeypatch.setattr(qi2labDataStore, "load_local_corrected_image", load_image)
    monkeypatch.setitem(
        zhuang_conversion.__globals__, "image_has_signal", lambda _: True
    )
    monkeypatch.setitem(zhuang_conversion.__globals__, "estimate_shading", estimate)
    zhuang_conversion(tmp_path, output_path=tmp_path / "qi2labdatastore")
    assert [len(projections) for projections in fits] == [2, 2, 2]
