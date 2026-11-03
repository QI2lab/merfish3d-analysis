from pathlib import Path

import numpy as np
import pytest
from ndstorage import NDTiffDataset
from scripts import process_ndtiff_positions as script
from tifffile import TiffFile, imread
from typer.testing import CliRunner


@pytest.fixture
def ndtiff_acquisition(tmp_path, request):
    """Write and reopen a two-position acquisition using ndstorage."""
    path = tmp_path / "acquisition"
    writer = NDTiffDataset(
        str(path), summary_metadata={"ChNames": ["DAPI", "Cy5"]}, writable=True
    )
    variant = getattr(request, "param", None)
    realistic = variant == "realistic"
    if realistic:
        from scipy.ndimage import gaussian_filter

    for channel in ("DAPI", "Cy5"):
        channel_coordinate = (
            ("DAPI", "Cy5").index(channel) if variant == "numeric" else channel
        )
        for position in (0, 2):
            if realistic:
                rng = np.random.default_rng(position + (100 if channel == "Cy5" else 0))
                beads = np.zeros((9, 96, 96), dtype=np.float32)
                for _ in range(24):
                    z, y, x = rng.integers([1, 8, 8], [8, 88, 88])
                    beads[z, y, x] += rng.uniform(1000, 3000)
                signal = gaussian_filter(beads, sigma=(0.8, 1.4, 1.4))
                raw = rng.poisson(5 + signal).astype(np.uint16) + 100
            for z in range(9 if realistic else 3):
                writer.put_image(
                    {
                        "channel": channel_coordinate,
                        "position": position,
                        "z": z,
                        "time": 0,
                    },
                    raw[z]
                    if realistic
                    else np.full(
                        (3, 4),
                        100 + (10 if channel == "Cy5" else 0) + position + z,
                        dtype=np.uint16,
                    ),
                    {
                        "Camera-CameraName": "C13440-20CU",
                        "Camera-CONVERSION FACTOR COEFF": "2.0",
                        "Camera-CONVERSION FACTOR OFFSET": "100.0",
                        "PixelSizeUm": "0.108",
                        "ZPosition_um_Intended": z * 0.315,
                    },
                )
    if variant == "multiple_times":
        writer.put_image(
            {"channel": "DAPI", "position": 0, "z": 0, "time": 1},
            np.full((3, 4), 100, dtype=np.uint16),
            {},
        )
    writer.finish()
    dataset = script._load_dataset(path)
    yield dataset
    dataset.close()


@pytest.mark.unit
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


@pytest.mark.unit
def test_generate_channel_psfs_uses_repository_microscope_parameters():
    from merfish3danalysis.utils.psf import generate_qi2lab_psf

    spacing = (0.32, 0.0985, 0.0985)
    psfs = script.generate_channel_psfs(
        ["Blue", "Yellow", "Red"], z_depth=7, voxel_size_zyx_um=spacing
    )
    for channel, wavelength in zip(psfs, (0.520, 0.580, 0.670), strict=True):
        expected = generate_qi2lab_psf(
            z_depth=7, voxel_size_zyx_um=spacing, emission_wavelength_um=wavelength
        )
        np.testing.assert_array_equal(psfs[channel], expected)
        assert psfs[channel].dtype == np.float32
        assert psfs[channel].sum() == pytest.approx(1)


@pytest.mark.unit
def test_list_channels_prints_metadata_names_without_processing(
    ndtiff_acquisition, tmp_path
):
    result = CliRunner().invoke(
        script.app, [str(tmp_path / "acquisition"), "--list-channels"]
    )

    assert result.exit_code == 0
    assert result.output.endswith(
        "Channels parsed from NDTiff metadata:\n- DAPI\n- Cy5\n"
    )


@pytest.mark.unit
@pytest.mark.parametrize("ndtiff_acquisition", ["numeric"], indirect=True)
def test_inspect_dataset_maps_summary_names_to_numeric_channels(ndtiff_acquisition):
    layout = script.inspect_dataset(ndtiff_acquisition)

    assert layout.channels == ("DAPI", "Cy5")
    assert layout.channel_coordinates == (0, 1)


@pytest.mark.unit
def test_aliases_missing_first_stack_to_zero_suffixed_reader(
    ndtiff_acquisition, tmp_path
):
    ndtiff_acquisition.close()
    path = tmp_path / "acquisition"
    (path / "NDTiffStack.tif").rename(path / "NDTiffStack_0.tif")
    dataset = script._load_dataset(path)
    try:
        np.testing.assert_array_equal(
            dataset.read_image(channel="DAPI", position=0, z=0, time=0),
            np.full((3, 4), 100, dtype=np.uint16),
        )
    finally:
        dataset.close()


@pytest.mark.integration
def test_exports_corrected_channels_and_selected_z_planes(ndtiff_acquisition, tmp_path):
    output = tmp_path / "output"
    script.process_dataset(
        ndtiff_acquisition,
        output_dir=output,
        ufish_channels=[],
        z_start=1,
        z_stop=3,
        overwrite=False,
        ufish=None,
    )
    for position in (0, 2):
        directory = output / f"pos{position:04d}"
        assert sorted(path.name for path in directory.iterdir()) == [
            "Cy5.ome.tif",
            "DAPI.ome.tif",
        ]
        for channel, offset in (("DAPI", 0), ("Cy5", 10)):
            expected = np.stack(
                [np.full((3, 4), 2 * (offset + position + z)) for z in (1, 2)]
            ).astype(np.uint16)
            with TiffFile(directory / f"{channel}.ome.tif") as tiff:
                np.testing.assert_array_equal(tiff.asarray(), expected)
                assert tiff.is_ome
                assert all(page.compression.name == "NONE" for page in tiff.pages)
                metadata = tiff.ome_metadata or ""
            assert 'PhysicalSizeX="0.108"' in metadata
            assert 'PhysicalSizeY="0.108"' in metadata
            assert 'PhysicalSizeZ="0.315"' in metadata
            assert f'Name="{channel}"' in metadata


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("ndtiff_acquisition", ["realistic"], indirect=True)
def test_cli_deconvolves_real_ndtiff_and_predicts_selected_channel(
    ndtiff_acquisition, tmp_path
):
    output = tmp_path / "output"
    result = CliRunner().invoke(
        script.app,
        [
            str(tmp_path / "acquisition"),
            "--output-dir",
            str(output),
            "--ufish-channel",
            "Cy5",
            "--deconvolve",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)

    for position in (0, 2):
        directory = output / f"pos{position:04d}"
        assert sorted(path.name for path in directory.iterdir()) == [
            "Cy5.ome.tif",
            "Cy5_decon.ome.tif",
            "Cy5_ufish.ome.tif",
            "DAPI.ome.tif",
            "DAPI_decon.ome.tif",
        ]
        for channel in ("DAPI", "Cy5"):
            raw = np.stack(
                [
                    ndtiff_acquisition.read_image(
                        channel=channel, position=position, z=z, time=0
                    )
                    for z in range(9)
                ]
            )
            corrected = imread(directory / f"{channel}.ome.tif")
            np.testing.assert_array_equal(corrected, (raw - 100) * 2)
            deconvolved = imread(directory / f"{channel}_decon.ome.tif")
            assert deconvolved.shape == corrected.shape
            assert deconvolved.dtype == np.uint16
            assert np.any(deconvolved)
            assert not np.array_equal(deconvolved, corrected)
            with TiffFile(directory / f"{channel}_decon.ome.tif") as tiff:
                assert tiff.is_ome
                metadata = tiff.ome_metadata or ""
            assert 'PhysicalSizeX="0.108"' in metadata
            assert 'PhysicalSizeY="0.108"' in metadata
            assert 'PhysicalSizeZ="0.315"' in metadata
        prediction = imread(directory / "Cy5_ufish.ome.tif")
        assert prediction.shape == (9, 96, 96)
        assert prediction.dtype == np.float32
        assert np.isfinite(prediction).all()
        assert prediction.max() > 0

    # Check the actual model received deconvolved rather than corrected pixels.
    model = script.load_default_ufish(0)
    _, expected = model.predict(
        imread(output / "pos0000" / "Cy5_decon.ome.tif"),
        axes="zyx",
        blend_3d=False,
        batch_size=1,
    )
    np.testing.assert_allclose(
        imread(output / "pos0000" / "Cy5_ufish.ome.tif"), expected, atol=1e-5
    )


@pytest.mark.unit
def test_process_dataset_preflights_existing_outputs(
    tmp_path: Path, ndtiff_acquisition
) -> None:
    dataset = ndtiff_acquisition
    existing_path = tmp_path / "output" / "pos0000" / "DAPI.ome.tif"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="--overwrite"):
        script.process_dataset(
            dataset,
            output_dir=tmp_path / "output",
            ufish_channels=["Cy5"],
            z_start=0,
            z_stop=1,
            overwrite=False,
            ufish=None,
        )

    assert existing_path.read_bytes() == b"keep me"


@pytest.mark.unit
def test_negative_one_z_stop_includes_final_plane(
    tmp_path: Path, ndtiff_acquisition
) -> None:
    script.process_dataset(
        ndtiff_acquisition,
        output_dir=tmp_path / "output",
        ufish_channels=[],
        z_start=0,
        z_stop=-1,
        overwrite=False,
        ufish=None,
    )

    corrected = imread(tmp_path / "output" / "pos0000" / "DAPI.ome.tif")
    assert corrected.shape == (3, 3, 4)
    np.testing.assert_array_equal(corrected[:, 0, 0], [0, 2, 4])


@pytest.mark.unit
def test_process_dataset_requires_exact_metadata_channel_name(
    tmp_path: Path, ndtiff_acquisition
) -> None:
    with pytest.raises(ValueError, match="Available channels"):
        script.process_dataset(
            ndtiff_acquisition,
            output_dir=tmp_path / "output",
            ufish_channels=["cy5"],
            z_start=0,
            z_stop=1,
            overwrite=False,
            ufish=None,
        )


@pytest.mark.unit
@pytest.mark.parametrize("ndtiff_acquisition", ["multiple_times"], indirect=True)
def test_inspect_dataset_rejects_non_singleton_time_axis(ndtiff_acquisition) -> None:
    with pytest.raises(ValueError, match="Axis 'time' has 2 values"):
        script.inspect_dataset(ndtiff_acquisition)


@pytest.mark.unit
def test_camera_correction_clips_to_uint16() -> None:
    raw = np.asarray([[-1, 100, 101, 40000]], dtype=np.int32)

    corrected = script.correct_camera_image(raw, gain=2.0, offset=100.0)

    np.testing.assert_array_equal(
        corrected,
        np.asarray([[0, 0, 2, 65535]], dtype=np.uint16),
    )
