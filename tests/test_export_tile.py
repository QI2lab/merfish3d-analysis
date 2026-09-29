"""Corrected tile export and exact local/global spot annotation."""

import sys
import types
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import tifffile
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import export_tile as exporter
from merfish3danalysis.utils import decode_warping


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
def export_setup(tmp_path, monkeypatch, spot_tables):
    store = tmp_path / "qi2labdatastore"
    store.mkdir()
    (store / "datastore_state.json").write_text("{}")
    native = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    datastore = Mock(
        tile_ids=["tile0000"],
        round_ids=["round001", "round002"],
        bit_ids=["bit001", "bit002"],
        voxel_size_zyx_um=[0.5, 0.2, 0.2],
    )
    datastore.load_local_corrected_image.return_value = native
    datastore.load_local_decoded_spots.return_value = spot_tables[0]
    datastore.load_global_filtered_decoded_spots.return_value = spot_tables[1]
    affine = np.eye(4, dtype=np.float32)
    affine[2, 3] = 0.2
    datastore.load_local_round_transform_zyx_um.return_value = affine
    datastore.load_local_sofima_flow_field.return_value = None
    datastore.load_local_round_linker.side_effect = lambda **kw: (
        1 if kw["bit"] == "bit001" else 2
    )
    chromatic = np.eye(4, dtype=np.float32)
    chromatic[1, 3] = 0.4
    datastore.load_chromatic_affine_transform_zyx_um.return_value = chromatic
    datastore.load_local_wavelengths_um.return_value = (0.6, 0.67)
    monkeypatch.setattr(exporter, "qi2labDataStore", Mock(return_value=datastore))
    warps = []

    def cpu_warp(image, *, transform_zyx_um, spacing_zyx_um, reference_shape, gpu_id):
        from scipy.ndimage import affine_transform

        warps.append((transform_zyx_um.copy(), tuple(reference_shape), gpu_id))
        spacing = np.asarray(spacing_zyx_um)
        matrix = transform_zyx_um[:3, :3] * spacing[None, :] / spacing[:, None]
        offset = transform_zyx_um[:3, 3] / spacing
        return affine_transform(
            image.astype(np.float32),
            matrix,
            offset,
            output_shape=reference_shape,
            order=1,
        )

    backend = types.ModuleType("merfish3danalysis.utils.multiview_registration")
    backend.warp_array_to_reference_gpu = cpu_warp
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    return store, datastore, native, affine, chromatic, warps


@pytest.mark.integration
@pytest.mark.parametrize("direct", [False, True])
def test_cli_exports_images_and_csv(export_setup, direct):
    store, datastore, native, affine, chromatic, warps = export_setup
    result = CliRunner().invoke(
        exporter.app,
        [
            str(store if direct else store.parent),
            "--tile-id",
            "tile0000",
            "--gpu-id",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    output = store.parent / "tile_exports" / "tile0000"
    for folder in ("native", "registered"):
        assert len(list((output / folder).glob("*.tif"))) == 4
    np.testing.assert_array_equal(tifffile.imread(output / "native/bit001.tif"), native)
    registered = tifffile.imread(output / "registered/fiducial_round001.tif")
    np.testing.assert_array_equal(registered, native)
    assert registered.dtype == np.float32
    # A +1 input-X translation samples native x=1 at reference x=0.
    shifted = tifffile.imread(output / "registered/fiducial_round002.tif")
    np.testing.assert_array_equal(shifted[:, :, 0], native[:, :, 1])
    with tifffile.TiffFile(output / "native/bit001.tif") as tif:
        assert tif.series[0].axes == "ZYX"
        assert 'PhysicalSizeZ="0.5"' in tif.ome_metadata
    exported = pd.read_csv(output / "decoded_spots.csv.gz")
    assert exported.passes_global_filter.tolist() == [True, False, True]
    assert pd.isna(exported.cell_id.iloc[1])
    np.testing.assert_allclose(warps[0][0], affine)
    np.testing.assert_allclose(warps[1][0], np.linalg.inv(chromatic))
    np.testing.assert_allclose(warps[2][0], np.linalg.inv(chromatic) @ affine)
    assert all(shape == native.shape and gpu == 2 for _, shape, gpu in warps)
    datastore.load_local_deconvolved_fiducial_image.assert_not_called()
    datastore.load_local_deconvolved_readout_image.assert_not_called()
    assert sorted(p.name for p in store.iterdir()) == ["datastore_state.json"]


@pytest.mark.integration
def test_cli_invalid_tile_and_output_paths(export_setup, tmp_path):
    store, *_ = export_setup
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
def test_failed_export_does_not_publish_partial_files(export_setup):
    store, datastore, *_ = export_setup
    datastore.load_local_round_transform_zyx_um.return_value = None
    with pytest.raises(ValueError, match="Missing local round transform"):
        exporter.export_tile(store, "tile0000", verbose=0)
    assert list((store.parent / "tile_exports").iterdir()) == []


@pytest.mark.integration
def test_overwrite_and_failed_overwrite_preserve_previous_output(export_setup):
    store, datastore, *_ = export_setup
    output = store.parent / "custom"
    exporter.export_tile(store, "tile0000", output_dir=output, verbose=0)
    before = (output / "decoded_spots.csv.gz").read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_tile(store, "tile0000", output_dir=output, verbose=0)
    datastore.load_local_wavelengths_um.return_value = None
    with pytest.raises(ValueError, match="wavelength"):
        exporter.export_tile(
            store, "tile0000", output_dir=output, overwrite=True, verbose=0
        )
    assert (output / "decoded_spots.csv.gz").read_bytes() == before
    datastore.load_local_wavelengths_um.return_value = (0.6, 0.67)
    exporter.export_tile(
        store, "tile0000", output_dir=output, overwrite=True, verbose=0
    )
    assert len(list((output / "registered").glob("*.tif"))) == 4


@pytest.mark.unit
@pytest.mark.parametrize("flow_status", [None, "identity_fallback_test", "success"])
def test_reference_shape_override_reaches_affine_and_flow_backends(
    monkeypatch, flow_status
):
    image = np.zeros((1, 3, 4), dtype=np.uint16)
    shape = (2, 4, 5)
    backend = types.ModuleType("merfish3danalysis.utils.multiview_registration")
    backend.warp_array_to_reference_gpu = Mock(return_value=np.zeros(shape))
    backend.warp_array_to_reference_with_affine_and_sofima_flow_gpu = Mock(
        return_value=np.zeros(shape)
    )
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    flow = (
        None
        if flow_status is None
        else (
            np.zeros((3, 1, 3, 4)),
            {
                "sofima_status": flow_status,
                "reference_shape_zyx_px": image.shape,
                "map_stride_zyx_px": [1, 1, 1],
                "map_box_start_xyz_px": [0, 0, 0],
            },
        )
    )
    warped = decode_warping.warp_image_to_reference_frame(
        image,
        transform_zyx_um=np.eye(4),
        spacing_zyx_um=[1, 1, 1],
        reference_shape=shape,
        loaded_flow_field=flow,
    )
    target = (
        backend.warp_array_to_reference_with_affine_and_sofima_flow_gpu
        if flow_status == "success"
        else backend.warp_array_to_reference_gpu
    )
    assert target.call_args.kwargs["reference_shape"] == shape
    assert warped.shape == shape
    assert warped.dtype == np.float32


@pytest.mark.integration
@pytest.mark.parametrize("missing", ["image", "local", "global"])
def test_missing_inputs_fail_before_output_publication(export_setup, missing):
    store, datastore, *_ = export_setup
    loader = {
        "image": datastore.load_local_corrected_image,
        "local": datastore.load_local_decoded_spots,
        "global": datastore.load_global_filtered_decoded_spots,
    }[missing]
    loader.return_value = None
    with pytest.raises(FileNotFoundError):
        exporter.export_tile(store, "tile0000", verbose=0)
    assert not (store.parent / "tile_exports" / "tile0000").exists()


@pytest.mark.integration
def test_2d_images_and_different_native_shape(export_setup):
    store, datastore, *_ = export_setup
    reference = np.arange(20, dtype=np.uint16).reshape(4, 5)

    def load_image(*, tile, return_future, round=None, bit=None):
        if round == "round001":
            return reference
        return np.ones((1, 3, 4), dtype=np.uint16)

    datastore.load_local_corrected_image.side_effect = load_image
    exporter.export_tile(store, "tile0000", verbose=0)
    output = store.parent / "tile_exports/tile0000"
    np.testing.assert_array_equal(
        tifffile.imread(output / "native/fiducial_round001.tif"), reference
    )
    assert tifffile.imread(output / "native/bit001.tif").shape == (3, 4)
    assert tifffile.imread(output / "registered/bit001.tif").shape == (4, 5)


@pytest.mark.integration
def test_fiducial_and_bit_receive_round_flow(export_setup, monkeypatch):
    store, datastore, native, *_ = export_setup
    flow = (
        np.zeros((3, *native.shape)),
        {
            "sofima_status": "success",
            "reference_shape_zyx_px": native.shape,
            "map_stride_zyx_px": [1, 1, 1],
            "map_box_start_xyz_px": [0, 0, 0],
        },
    )
    datastore.load_local_sofima_flow_field.return_value = flow
    backend = sys.modules["merfish3danalysis.utils.multiview_registration"]
    flow_warp = Mock(return_value=np.full(native.shape, 0.5, dtype=np.float32))
    monkeypatch.setattr(
        backend,
        "warp_array_to_reference_with_affine_and_sofima_flow_gpu",
        flow_warp,
        raising=False,
    )
    exporter.export_tile(store, "tile0000", verbose=0)
    assert flow_warp.call_count == 2
    for call in flow_warp.call_args_list:
        assert call.args[0].dtype == np.float32
        assert call.kwargs["sofima_flow_field_xyz_px"] is flow[0]
        assert call.kwargs["reference_shape"] == native.shape
    output = store.parent / "tile_exports/tile0000"
    np.testing.assert_array_equal(
        tifffile.imread(output / "registered/bit002.tif"), np.full(native.shape, 0.5)
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("with_flow", [False, True])
def test_export_preserves_fractional_intensities_on_gpu(
    export_setup, monkeypatch, with_flow
):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")
    store, datastore, native, *_ = export_setup
    monkeypatch.delitem(sys.modules, "merfish3danalysis.utils.multiview_registration")
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = 0.1  # Half an X pixel at 0.2 um per pixel.
    datastore.load_local_round_transform_zyx_um.return_value = transform
    if with_flow:
        flow = np.zeros((3, *native.shape), dtype=np.float32)
        flow[0] = 0.25  # Add a quarter X pixel before the affine transform.
        datastore.load_local_sofima_flow_field.return_value = (
            flow,
            {
                "sofima_status": "success",
                "reference_shape_zyx_px": native.shape,
                "map_stride_zyx_px": [1, 1, 1],
                "map_box_start_xyz_px": [0, 0, 0],
            },
        )
    exporter.export_tile(store, "tile0000", verbose=0)
    output = store.parent / "tile_exports/tile0000/registered/fiducial_round002.tif"
    registered = tifffile.imread(output)
    assert registered.dtype == np.float32
    assert registered.shape == native.shape
    np.testing.assert_allclose(
        registered[:, :, 0], native[:, :, 0] + (0.75 if with_flow else 0.5), atol=1e-5
    )
