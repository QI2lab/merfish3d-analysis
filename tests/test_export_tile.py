"""Corrected tile export and exact local/global spot annotation."""

import numpy as np
import pandas as pd
import pytest
import tifffile
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import export_tile as exporter


@pytest.fixture
def spot_tables():
    local = pd.DataFrame(
        {
            "tile_idx": [0, 0, 0],
            "gene_id": ["A", "A", "B"],
            "barcode_id": [1, 1, 2],
            "z": [0.0, 1.0, 2.0],
            "y": [2.5, 2.5, 4.0],
            "x": [3.0, 3.0, 5.0],
            "distance_min": [0.1, 0.2, 0.3],
        },
        index=[8, 2, 9],
    )
    global_spots = local.iloc[[2, 0]].copy().reset_index(drop=True)
    global_spots["blank_fraction"] = [0.02, 0.01]
    global_spots["cell_id"] = [10, 20]
    global_spots.loc[0, "distance_min"] = 0.35
    other_tile = global_spots.iloc[[0]].assign(tile_idx=1)
    return local, pd.concat([global_spots, other_tile], ignore_index=True)


@pytest.mark.unit
def test_annotation_preserves_rows_and_all_statistics(spot_tables):
    local, global_spots = spot_tables
    result = exporter.annotate_local_spots(local, global_spots, 0)
    pd.testing.assert_frame_equal(result[local.columns], local)
    assert result.passes_global_filter.tolist() == [True, False, True]
    assert result.cell_id.iloc[[0, 2]].tolist() == [20, 10]
    assert result.blank_fraction.iloc[[0, 2]].tolist() == [0.01, 0.02]
    assert result.global_file_distance_min.iloc[2] == 0.35
    assert pd.isna(result.cell_id.iloc[1])
    assert "passes_global_filter" not in local


@pytest.mark.unit
@pytest.mark.parametrize("empty_local", [False, True])
def test_empty_tables(spot_tables, empty_local):
    local, global_spots = spot_tables
    if empty_local:
        local = local.iloc[:0]
    result = exporter.annotate_local_spots(local, global_spots.iloc[:0], 0)
    assert len(result) == len(local)
    assert not result.passes_global_filter.any()
    assert result.cell_id.isna().all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_local",
        "duplicate_global",
        "unmatched",
        "missing",
        "null",
        "collision",
        "wrong_tile",
    ],
)
def test_annotation_rejects_ambiguous_or_inconsistent_tables(spot_tables, problem):
    local, global_spots = spot_tables
    if problem == "duplicate_local":
        local = pd.concat([local, local.iloc[[0]]])
    elif problem == "duplicate_global":
        global_spots = pd.concat([global_spots, global_spots.iloc[[0]]])
    elif problem == "unmatched":
        global_spots.loc[0, "x"] += 0.001
    elif problem == "missing":
        local = local.drop(columns="z")
    elif problem == "null":
        local.loc[8, "x"] = np.nan
    elif problem == "collision":
        local["global_file_distance_min"] = 1.0
    else:
        local.loc[8, "tile_idx"] = 2
    with pytest.raises(ValueError):
        exporter.annotate_local_spots(local, global_spots, 0)


@pytest.fixture
def export_setup(image_datastore, spot_tables, request):
    """Populate the shared acquisition with corrected images and decoded spots."""
    datastore = image_datastore
    native = np.arange(8 * 12 * 16, dtype=np.uint16).reshape(8, 12, 16)
    for round_id in datastore.round_ids:
        datastore.save_local_corrected_image(native, tile=0, round=round_id)
    for bit in datastore.bit_ids:
        if getattr(request, "param", None) == "missing_image" and bit == "bit002":
            continue
        image = (
            np.ones((1, 6, 8), dtype=np.uint16)
            if getattr(request, "param", None) == "different_shape" and bit == "bit001"
            else native
        )
        datastore.save_local_corrected_image(image, tile=0, bit=bit)
    datastore.save_local_decoded_spots(spot_tables[0], tile=0)
    datastore.save_global_filtered_decoded_spots(spot_tables[1])
    affine = np.eye(4, dtype=np.float32)
    affine[2, 3] = 0.4
    datastore.save_local_round_transform_zyx_um(affine, tile=0, round=1)
    chromatic = np.eye(4, dtype=np.float32)
    chromatic[1, 3] = 0.4
    datastore.save_chromatic_affine_transforms_zyx_um(
        {
            "estimator": "decoded_rna_on_bit_weighted_centroid_z_translation_yx_affine_graph",
            "channels": {
                "readout1": {
                    "wavelength_um": 0.67,
                    "affine_zyx_um": chromatic,
                    "status": "ok",
                }
            },
        }
    )
    return datastore, native


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("direct", [False, True])
def test_cli_exports_images_and_csv(export_setup, direct):
    datastore, native = export_setup
    store = datastore.datastore_path
    result = CliRunner().invoke(
        exporter.app,
        [str(store if direct else store.parent), "--tile-id", "tile0000"],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    output = store.parent / "tile_exports" / "tile0000"
    for folder in ("native", "registered"):
        assert len(list((output / folder).glob("*.tif"))) == 6
    np.testing.assert_array_equal(tifffile.imread(output / "native/bit001.tif"), native)
    registered = tifffile.imread(output / "registered/fiducial_round001.tif")
    np.testing.assert_array_equal(registered, native)
    assert registered.dtype == np.float32
    shifted = tifffile.imread(output / "registered/fiducial_round002.tif")
    np.testing.assert_array_equal(shifted[:, :, :-1], native[:, :, 1:])
    # Chromatic corrections belong to decoding and do not change this export.
    np.testing.assert_array_equal(
        tifffile.imread(output / "registered/bit001.tif"), native
    )
    np.testing.assert_array_equal(
        tifffile.imread(output / "registered/bit002.tif")[:, :, :-1], native[:, :, 1:]
    )
    with tifffile.TiffFile(output / "native/bit001.tif") as tif:
        assert tif.series[0].axes == "ZYX"
        assert 'PhysicalSizeZ="0.4"' in tif.ome_metadata
    exported = pd.read_csv(output / "decoded_spots.csv.gz")
    assert exported.passes_global_filter.tolist() == [True, False, True]
    assert pd.isna(exported.cell_id.iloc[1])


@pytest.mark.integration
def test_cli_invalid_tile_and_output_paths(export_setup, tmp_path):
    datastore, _ = export_setup
    store = datastore.datastore_path
    runner = CliRunner()
    for arguments in (
        ["--tile-id", "bad"],
        ["--tile-id", "tile0000", "--output-dir", str(store / "exports")],
    ):
        assert runner.invoke(exporter.app, [str(store), *arguments]).exit_code != 0
    link = tmp_path / "linked_output"
    link.symlink_to(store, target_is_directory=True)
    assert (
        runner.invoke(
            exporter.app,
            [str(store), "--tile-id", "tile0000", "--output-dir", str(link)],
        ).exit_code
        != 0
    )
    assert not (store.parent / "tile_exports").exists()


@pytest.mark.integration
@pytest.mark.gpu
def test_overwrite_and_failed_overwrite_preserve_previous_output(
    export_setup, spot_tables
):
    datastore, _ = export_setup
    store = datastore.datastore_path
    output = store.parent / "custom"
    exporter.export_tile(store, "tile0000", output_dir=output, verbose=0)
    before = (output / "decoded_spots.csv.gz").read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_tile(store, "tile0000", output_dir=output, verbose=0)
    datastore.save_local_decoded_spots(
        pd.concat([spot_tables[0], spot_tables[0]]), tile=0
    )
    with pytest.raises(ValueError, match="duplicate"):
        exporter.export_tile(
            store, "tile0000", output_dir=output, overwrite=True, verbose=0
        )
    assert (output / "decoded_spots.csv.gz").read_bytes() == before
    datastore.save_local_decoded_spots(spot_tables[0], tile=0)
    exporter.export_tile(
        store, "tile0000", output_dir=output, overwrite=True, verbose=0
    )
    assert len(list((output / "registered").glob("*.tif"))) == 6


@pytest.mark.integration
@pytest.mark.parametrize("missing", ["image", "local", "global"])
def test_missing_inputs_fail_before_output_publication(
    image_datastore, spot_tables, missing
):
    datastore = image_datastore
    if missing != "image":
        datastore.save_local_corrected_image(
            np.ones((4, 8, 8), dtype=np.uint16), tile=0, round=0
        )
    if missing != "local":
        datastore.save_local_decoded_spots(spot_tables[0], tile=0)
    if missing != "global":
        datastore.save_global_filtered_decoded_spots(spot_tables[1])
    with pytest.raises(FileNotFoundError):
        exporter.export_tile(datastore.datastore_path, "tile0000", verbose=0)
    assert not (datastore.datastore_path.parent / "tile_exports" / "tile0000").exists()


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("export_setup", ["missing_image"], indirect=True)
def test_missing_later_image_does_not_publish_partial_files(export_setup):
    datastore, _ = export_setup
    store = datastore.datastore_path
    with pytest.raises(FileNotFoundError, match="bit002"):
        exporter.export_tile(store, "tile0000", verbose=0)
    assert list((store.parent / "tile_exports").iterdir()) == []


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("export_setup", ["different_shape"], indirect=True)
def test_bit_is_warped_to_reference_shape(export_setup):
    datastore, native = export_setup
    store = datastore.datastore_path
    exporter.export_tile(store, "tile0000", verbose=0)
    output = store.parent / "tile_exports/tile0000"
    assert tifffile.imread(output / "native/bit001.tif").shape == (6, 8)
    registered = tifffile.imread(output / "registered/bit001.tif")
    assert registered.shape == native.shape
    np.testing.assert_array_equal(registered[0, :6, :8], 1)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("with_flow", [False, True])
def test_export_preserves_fractional_intensities_on_gpu(export_setup, with_flow):
    datastore, native = export_setup
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = 0.2  # Half an X pixel at 0.4 um per pixel.
    datastore.save_local_round_transform_zyx_um(transform, tile=0, round=1)
    if with_flow:
        flow = np.zeros((3, *native.shape), dtype=np.float32)
        flow[0] = 0.25
        datastore.save_local_sofima_flow_field(
            flow,
            tile=0,
            round=1,
            reference_round=0,
            map_stride_zyx_px=(1, 1, 1),
            map_box_start_xyz_px=(0, 0, 0),
            map_box_size_xyz_px=native.shape[::-1],
            reference_shape_zyx_px=native.shape,
            moving_shape_zyx_px=native.shape,
            sofima_status="success",
        )
    store = datastore.datastore_path
    exporter.export_tile(store, "tile0000", verbose=0)
    output = store.parent / "tile_exports/tile0000/registered"
    expected = native[:, :, :-1] + (0.75 if with_flow else 0.5)
    for image_id in ("fiducial_round002", "bit002"):
        registered = tifffile.imread(output / f"{image_id}.tif")
        assert registered.dtype == np.float32
        assert registered.shape == native.shape
        np.testing.assert_allclose(registered[:, :, :-1], expected, atol=1e-4)
