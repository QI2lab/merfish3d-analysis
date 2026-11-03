"""Zhuang illumination correction with generated images and real persistence."""

import json
import runpy
import weakref
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from tifffile import TiffFile, imread, imwrite
from yaozarrs import open_group

from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.fixture
def zhuang_conversion(tmp_path):
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
    additional = tmp_path / "additional_files"
    positions = additional / "fov_positions"
    positions.mkdir(parents=True)
    codebook.to_csv(additional / "codebook.csv", index=False)
    pd.DataFrame([[0, 1, 2], [3, 4, 5]]).to_csv(
        positions / "mouse1_sample1.txt", index=False, header=False
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
    images = tmp_path / "mouse1_sample1_raw"
    images.mkdir()
    imwrite(images / "tile0.tif", raw, photometric="minisblack")
    imwrite(images / "tile1.tif", second, photometric="minisblack")
    return convert


@pytest.mark.integration
@pytest.mark.gpu
def test_zhuang_conversion_estimates_and_applies_real_flatfields(
    zhuang_conversion, tmp_path
):
    """Convert varied TIFF tiles through signal screening and real BaSiC fitting."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("basicpy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")

    signal_tiles = 6
    num_tiles = signal_tiles + 1
    positions = np.column_stack(
        (np.arange(num_tiles) * 10, np.arange(num_tiles) * 20, np.zeros(num_tiles))
    )
    pd.DataFrame(positions).to_csv(
        tmp_path / "additional_files/fov_positions/mouse1_sample1.txt",
        index=False,
        header=False,
    )
    y, x = np.mgrid[-1:1:64j, -1:1:96j]
    expected_flatfields = np.stack(
        [1 - (0.16 + 0.02 * channel) * y**2 - 0.12 * x**2 for channel in range(3)]
    )
    expected_flatfields /= expected_flatfields.max(axis=(1, 2), keepdims=True)
    raw_dir = tmp_path / "mouse1_sample1_raw"
    for tile in range(num_tiles):
        raw = np.empty((39, 2, 64, 96), dtype=np.uint16)
        spot = np.exp(
            -((y - (-0.45 + 0.15 * tile)) ** 2 + (x - (0.35 - 0.1 * tile)) ** 2)
            / (2 * 0.045**2)
        )
        for channel in range(39):
            color = 0 if channel == 38 else 1 + channel % 2
            intensity = 800 + 90 * tile + 15 * (channel % 5)
            image = intensity * expected_flatfields[color] + 1500 * spot
            if channel == 38 and tile == signal_tiles:
                image = np.full((64, 96), intensity)
            raw[channel, 0] = 100 + 0.7 * image / 0.46
            raw[channel, 1] = 100 + image / 0.46
        imwrite(raw_dir / f"tile{tile}.tif", raw, photometric="minisblack")

    output = tmp_path / "qi2labdatastore"
    with pytest.warns(UserWarning, match="fiducial tiles contain signal"):
        zhuang_conversion(tmp_path, output_path=output)

    datastore = qi2labDataStore(output)
    assert datastore.datastore_state["Corrected"] is True
    assert json.loads(datastore.shading_flatfield_tile_ids_path().read_text()) == list(
        datastore.tile_ids[:signal_tiles]
    )
    with TiffFile(tmp_path / "illuminations.ome.tif") as tif:
        flatfields = tif.asarray()
        assert tif.series[0].axes == "CYX"
    assert flatfields.shape == (3, 96, 64)
    assert np.all(np.isfinite(flatfields)) and np.all(flatfields > 0)
    np.testing.assert_allclose(
        flatfields, expected_flatfields.transpose(0, 2, 1), rtol=0, atol=0.12
    )

    for tile in (0, signal_tiles):
        raw = imread(raw_dir / f"tile{tile}.tif")
        origin, camera = datastore.load_local_stage_position_zyx_um(tile, 0)
        np.testing.assert_array_equal(origin, positions[tile])
        np.testing.assert_array_equal(camera, np.eye(4))
        for kind, index, raw_channel, color in (
            ("round", 0, 38, 0),
            ("round", 18, 38, 0),
            ("bit", 0, 0, 1),
            ("bit", 1, 1, 2),
            ("bit", 37, 37, 2),
        ):
            selection = {kind: index}
            camera_corrected = (
                np.maximum(raw[raw_channel].astype(np.float32) - 100, 0) * 0.46
            ).astype(np.uint16)
            expected = (
                (camera_corrected.swapaxes(1, 2).astype(np.float32) / flatfields[color])
                .clip(0, 65535)
                .astype(np.uint16)
            )
            actual = datastore.load_local_corrected_image(
                tile=tile, return_future=False, **selection
            )
            np.testing.assert_array_equal(actual, expected)
            path = datastore.local_image_path(tile, "corrected_data", **selection)
            metadata = datastore.load_image_metadata(path)
            assert metadata["gain_correction"] and metadata["shading_correction"]
            assert metadata["psf_idx"] == color


@pytest.mark.unit
@pytest.mark.parametrize("max_images", [1, 2, 100])
def test_zhuang_illumination_corrects_and_persists_each_channel(
    zhuang_conversion, tmp_path, monkeypatch, max_images
):
    monkeypatch.setitem(
        zhuang_conversion.__globals__, "image_has_signal", Mock(return_value=True)
    )
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
    assert json.loads(datastore.shading_flatfield_tile_ids_path().read_text()) == list(
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
                multiscale = group.ome_metadata(strict=True).multiscales[0]
                np.testing.assert_allclose(
                    multiscale.datasets[0].transform.scale, [1.5, 0.108, 0.108]
                )

    with TiffFile(tmp_path / "illuminations.ome.tif") as tif:
        np.testing.assert_array_equal(tif.asarray(), flatfields)
        assert tif.series[0].axes == "CYX"
        pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        for axis in "XY":
            assert float(pixels.attrib[f"PhysicalSize{axis}"]) == pytest.approx(0.108)
            assert pixels.attrib[f"PhysicalSize{axis}Unit"] == "µm"


@pytest.mark.unit
def test_zhuang_failed_estimation_does_not_mark_conversion_complete(
    zhuang_conversion, tmp_path, monkeypatch
):
    monkeypatch.setitem(
        zhuang_conversion.__globals__, "image_has_signal", Mock(return_value=True)
    )
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


@pytest.mark.unit
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
    assert json.loads(datastore.shading_flatfield_tile_ids_path().read_text()) == [
        datastore.tile_ids[1]
    ]
    for tile in range(2):
        for kind in ("round", "bit"):
            path = datastore.local_image_path(tile, "corrected_data", **{kind: 0})
            assert datastore.load_image_metadata(path)["shading_correction"] is True


@pytest.mark.unit
def test_zhuang_no_signal_stops_before_flatfield_fitting(zhuang_conversion, tmp_path):
    for tile in range(2):
        imwrite(
            tmp_path / "mouse1_sample1_raw" / f"tile{tile}.tif",
            np.full((39, 2, 3, 4), 200, dtype=np.uint16),
            photometric="minisblack",
        )
    output = tmp_path / "qi2labdatastore"
    with pytest.raises(ValueError, match="No fiducial tiles"):
        zhuang_conversion(tmp_path, output_path=output, max_flatfield_images=1)
    datastore = qi2labDataStore(output, validate=False)
    assert json.loads(datastore.shading_flatfield_tile_ids_path().read_text()) == []
    assert datastore.datastore_state["Corrected"] is False


@pytest.mark.unit
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
