import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import process_ndtiff_positions as script
from tifffile import TiffFile, imread
from typer.testing import CliRunner


class _FakeDataset:
    def __init__(self) -> None:
        self.channels = ("DAPI", "Cy5")
        self.positions = (0, 2)
        self.z_values = (0, 1, 2)
        self.closed = False
        self.read_calls: list[dict[str, int | str]] = []
        self.coordinates = [
            {"channel": channel, "position": position, "z": z_value, "time": 0}
            for channel in self.channels
            for position in self.positions
            for z_value in self.z_values
        ]

    def get_image_coordinates_list(self) -> list[dict[str, int | str]]:
        return self.coordinates

    def get_channel_names(self) -> list[str]:
        return list(self.channels)

    def has_image(
        self,
        channel: str,
        position: int,
        z: int,
        time: int,
    ) -> bool:
        return {
            "channel": channel,
            "position": position,
            "z": z,
            "time": time,
        } in self.coordinates

    def read_image(
        self,
        channel: str,
        position: int,
        z: int,
        time: int,
    ) -> np.ndarray:
        coordinates = {
            "channel": channel,
            "position": position,
            "z": z,
            "time": time,
        }
        self.read_calls.append(coordinates)
        channel_value = 10 if channel == "Cy5" else 0
        return np.full(
            (3, 4),
            100 + channel_value + position + z,
            dtype=np.uint16,
        )

    def read_metadata(self, **coordinates: int | str) -> dict[str, object]:
        return {
            "Camera-CameraName": "C13440-20CU",
            "Camera-CONVERSION FACTOR COEFF": "2.0",
            "Camera-CONVERSION FACTOR OFFSET": "100.0",
            "PixelSizeUm": "0.108",
            "ZPosition_um_Intended": float(coordinates["z"]) * 0.315,
        }

    def close(self) -> None:
        self.closed = True


class _FakeUFish:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: str,
        blend_3d: bool,
        batch_size: int,
    ) -> tuple[None, np.ndarray]:
        assert axes == "zyx"
        assert blend_3d is False
        assert batch_size == 1
        self.calls.append(image.copy())
        return None, image.astype(np.float64) / 10.0


def test_expands_consecutive_ufish_channel_arguments() -> None:
    arguments = [
        "/data/acquisition",
        "--z-start",
        "5",
        "--ufish-channel",
        "Yellow",
        "Red",
        "--output-dir",
        "/data/output",
    ]

    assert script._expand_ufish_channel_arguments(arguments) == [
        "/data/acquisition",
        "--z-start",
        "5",
        "--ufish-channel",
        "Yellow",
        "--ufish-channel",
        "Red",
        "--output-dir",
        "/data/output",
    ]


def test_generate_channel_psfs_uses_repository_microscope_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_make_psf(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.ones((int(kwargs["z"]), 3, 3), dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "psfmodels",
        SimpleNamespace(make_psf=fake_make_psf),
    )

    psfs = script.generate_channel_psfs(
        ["Blue", "Yellow", "Red"],
        z_depth=7,
        voxel_size_zyx_um=(0.32, 0.0985, 0.0985),
    )

    assert [call["wvl"] for call in calls] == [0.520, 0.580, 0.670]
    for call in calls:
        assert call == {
            "z": 7,
            "nx": 51,
            "dxy": 0.0985,
            "dz": 0.32,
            "NA": 1.35,
            "wvl": call["wvl"],
            "ns": 1.47,
            "ni": 1.51,
            "ni0": 1.51,
            "model": "vectorial",
        }
    for psf in psfs.values():
        assert psf.dtype == np.float32
        assert np.isclose(np.sum(psf), 1.0)


def test_deconvolve_stack_uses_repository_rlgc_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_chunked_rlgc(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        return np.asarray([[[-1.0, 1.9, 70000.0]]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "merfish3danalysis.utils.rlgc",
        SimpleNamespace(chunked_rlgc=fake_chunked_rlgc),
    )
    image = np.ones((1, 2, 3), dtype=np.uint16)
    psf = np.ones((1, 3, 3), dtype=np.float32)

    deconvolved = script.deconvolve_stack(image, psf, gpu_id=2)

    assert len(calls) == 1
    assert calls[0]["image"] is image
    assert calls[0]["psf"] is psf
    assert calls[0]["gpu_id"] == 2
    assert calls[0]["crop_yx"] == 2048
    assert calls[0]["crop_z"] is None
    assert calls[0]["release_memory"] is True
    assert deconvolved.dtype == np.uint16
    np.testing.assert_array_equal(deconvolved, [[[0, 1, 65535]]])


def test_list_channels_prints_metadata_names_without_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _FakeDataset()
    monkeypatch.setattr(script, "_load_dataset", lambda _path: dataset)

    result = CliRunner().invoke(script.app, [str(tmp_path), "--list-channels"])

    assert result.exit_code == 0
    assert result.output == ("Channels parsed from NDTiff metadata:\n- DAPI\n- Cy5\n")
    assert dataset.read_calls == []
    assert dataset.closed


def test_inspect_dataset_maps_summary_names_to_numeric_channels() -> None:
    dataset = _FakeDataset()
    dataset.coordinates = [
        {**coordinate, "channel": dataset.channels.index(str(coordinate["channel"]))}
        for coordinate in dataset.coordinates
    ]
    dataset.get_channel_names = lambda: []
    dataset.summary_metadata = {"ChNames": ["DAPI", "Cy5"]}

    layout = script.inspect_dataset(dataset)

    assert layout.channels == ("DAPI", "Cy5")
    assert layout.channel_coordinates == (0, 1)


def test_aliases_missing_first_stack_to_zero_suffixed_reader() -> None:
    first_reader = object()
    dataset = SimpleNamespace(
        _readers_by_filename={"sample_NDTiffStack_0.tif": first_reader},
        index={0: {"filename": "sample_NDTiffStack.tif"}},
    )

    script._alias_zero_suffixed_stack_reader(dataset)

    assert dataset._readers_by_filename["sample_NDTiffStack.tif"] is first_reader


def test_process_dataset_writes_all_corrected_and_selected_ufish_stacks(
    tmp_path: Path,
) -> None:
    dataset = _FakeDataset()
    ufish = _FakeUFish()

    script.process_dataset(
        dataset,
        output_dir=tmp_path,
        ufish_channels=["Cy5"],
        z_start=1,
        z_stop=3,
        overwrite=False,
        ufish=ufish,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["pos0000", "pos0002"]
    assert sorted(path.name for path in (tmp_path / "pos0000").iterdir()) == [
        "Cy5.ome.tif",
        "Cy5_ufish.ome.tif",
        "DAPI.ome.tif",
    ]
    assert sorted(path.name for path in (tmp_path / "pos0002").iterdir()) == [
        "Cy5.ome.tif",
        "Cy5_ufish.ome.tif",
        "DAPI.ome.tif",
    ]

    expected_dapi = np.stack(
        [
            np.full((3, 4), 2, dtype=np.uint16),
            np.full((3, 4), 4, dtype=np.uint16),
        ]
    )
    expected_cy5 = expected_dapi + 20
    np.testing.assert_array_equal(
        imread(tmp_path / "pos0000" / "DAPI.ome.tif"), expected_dapi
    )
    np.testing.assert_array_equal(
        imread(tmp_path / "pos0000" / "Cy5.ome.tif"), expected_cy5
    )
    ufish_output = imread(tmp_path / "pos0000" / "Cy5_ufish.ome.tif")
    assert ufish_output.dtype == np.float32
    np.testing.assert_array_equal(
        ufish_output,
        expected_cy5.astype(np.float32) / 10.0,
    )
    with TiffFile(tmp_path / "pos0000" / "DAPI.ome.tif") as tiff:
        assert tiff.is_ome
        assert all(page.compression.name == "NONE" for page in tiff.pages)
        ome_metadata = tiff.ome_metadata or ""
    assert 'PhysicalSizeX="0.108"' in ome_metadata
    assert 'PhysicalSizeY="0.108"' in ome_metadata
    assert 'PhysicalSizeZ="0.315"' in ome_metadata
    assert 'Name="DAPI"' in ome_metadata
    assert len(ufish.calls) == 2


def test_process_dataset_preflights_existing_outputs(tmp_path: Path) -> None:
    dataset = _FakeDataset()
    existing_path = tmp_path / "pos0000" / "DAPI.ome.tif"
    existing_path.parent.mkdir()
    existing_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="--overwrite"):
        script.process_dataset(
            dataset,
            output_dir=tmp_path,
            ufish_channels=["Cy5"],
            z_start=0,
            z_stop=1,
            overwrite=False,
            ufish=_FakeUFish(),
        )

    assert existing_path.read_bytes() == b"keep me"
    assert dataset.read_calls == []


def test_deconvolution_writes_stacks_and_supplies_ufish_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ufish = _FakeUFish()
    deconvolution_calls = []
    psfs = {
        "DAPI": np.ones((2, 3, 3), dtype=np.float32),
        "Cy5": np.full((2, 3, 3), 2, dtype=np.float32),
    }
    monkeypatch.setattr(
        script,
        "generate_channel_psfs",
        lambda *_args, **_kwargs: psfs,
    )

    def fake_deconvolve(
        image: np.ndarray,
        psf: np.ndarray,
        *,
        gpu_id: int,
    ) -> np.ndarray:
        deconvolution_calls.append((image.copy(), psf, gpu_id))
        return image + 5

    monkeypatch.setattr(script, "deconvolve_stack", fake_deconvolve)

    script.process_dataset(
        _FakeDataset(),
        output_dir=tmp_path,
        ufish_channels=["Cy5"],
        z_start=0,
        z_stop=2,
        overwrite=False,
        ufish=ufish,
        gpu_id=3,
        deconvolve=True,
    )

    assert len(deconvolution_calls) == 4
    assert all(call[2] == 3 for call in deconvolution_calls)
    assert sorted(path.name for path in (tmp_path / "pos0000").iterdir()) == [
        "Cy5.ome.tif",
        "Cy5_decon.ome.tif",
        "Cy5_ufish.ome.tif",
        "DAPI.ome.tif",
        "DAPI_decon.ome.tif",
    ]
    deconvolved_cy5 = imread(tmp_path / "pos0000" / "Cy5_decon.ome.tif")
    np.testing.assert_array_equal(ufish.calls[0], deconvolved_cy5)
    for filename in ("Cy5_decon.ome.tif", "Cy5_ufish.ome.tif"):
        with TiffFile(tmp_path / "pos0000" / filename) as tiff:
            assert tiff.is_ome
            ome_metadata = tiff.ome_metadata or ""
        assert 'PhysicalSizeX="0.108"' in ome_metadata
        assert 'PhysicalSizeY="0.108"' in ome_metadata
        assert 'PhysicalSizeZ="0.315"' in ome_metadata


def test_negative_one_z_stop_includes_final_plane(tmp_path: Path) -> None:
    script.process_dataset(
        _FakeDataset(),
        output_dir=tmp_path,
        ufish_channels=["Cy5"],
        z_start=0,
        z_stop=-1,
        overwrite=False,
        ufish=_FakeUFish(),
    )

    corrected = imread(tmp_path / "pos0000" / "DAPI.ome.tif")
    assert corrected.shape == (3, 3, 4)
    np.testing.assert_array_equal(corrected[:, 0, 0], [0, 2, 4])


def test_process_dataset_requires_exact_metadata_channel_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Available channels"):
        script.process_dataset(
            _FakeDataset(),
            output_dir=tmp_path,
            ufish_channels=["cy5"],
            z_start=0,
            z_stop=1,
            overwrite=False,
            ufish=_FakeUFish(),
        )


def test_inspect_dataset_rejects_non_singleton_time_axis() -> None:
    dataset = _FakeDataset()
    dataset.coordinates.append({"channel": "DAPI", "position": 0, "z": 0, "time": 1})

    with pytest.raises(ValueError, match="Axis 'time' has 2 values"):
        script.inspect_dataset(dataset)


def test_camera_correction_clips_to_uint16() -> None:
    raw = np.asarray([[-1, 100, 101, 40000]], dtype=np.int32)

    corrected = script.correct_camera_image(raw, gain=2.0, offset=100.0)

    np.testing.assert_array_equal(
        corrected,
        np.asarray([[0, 0, 2, 65535]], dtype=np.uint16),
    )
