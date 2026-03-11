from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorstore as ts

from merfish3danalysis.qi2labDataStore import qi2labDataStore


def _write_legacy_tensorstore_array(path: Path, array: np.ndarray) -> None:
    compressor = {
        "id": "blosc",
        "cname": "zstd",
        "clevel": 5,
        "shuffle": 2,
    }
    if array.ndim == 3:
        chunks = [1, int(array.shape[1]), int(array.shape[2])]
    elif array.ndim == 2:
        chunks = [int(array.shape[0]), int(array.shape[1])]
    else:
        chunks = [int(v) for v in array.shape]

    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "create": True,
        "open": False,
        "delete_existing": True,
        "metadata": {
            "shape": [int(v) for v in array.shape],
            "chunks": chunks,
            "dtype": np.dtype(array.dtype).str,
            "compressor": compressor,
        },
    }
    legacy_array = ts.open(spec).result()
    legacy_array.write(array).result()


def _make_calibrated_store(
    datastore_path: Path, *, ragged_psfs: bool = False
) -> qi2labDataStore:
    datastore = qi2labDataStore(datastore_path)
    datastore.channels_in_data = [0, 1, 2]
    datastore.experiment_order = np.asarray([[1, 1, 0]], dtype=np.int64)
    datastore.num_tiles = 1
    datastore.tile_overlap = 0.1
    datastore.binning = 1
    datastore.e_per_ADU = 1.0
    datastore.na = 1.35
    datastore.ri = 1.47
    datastore.microscope_type = "3D"
    datastore.camera_model = "test-camera"
    datastore.voxel_size_zyx_um = [0.5, 0.2, 0.2]
    datastore.codebook = pd.DataFrame(
        [["gene_a", 1], ["gene_b", 0]], columns=["gene_id", "bit01"]
    )

    if ragged_psfs:
        datastore.channel_psfs = [
            np.ones((3, 5, 5), dtype=np.float32),
            np.ones((5, 7, 7), dtype=np.float32),
            np.ones((1, 3, 3), dtype=np.float32),
        ]
    else:
        datastore.channel_psfs = np.ones((3, 3, 5, 5), dtype=np.float32)

    state = dict(datastore.datastore_state)
    state["Calibrations"] = True
    datastore.datastore_state = state
    return datastore


def _populate_minimal_local_registered_store(datastore: qi2labDataStore) -> None:
    datastore.initialize_tile(0)

    stage_zyx_um = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    affine_zyx_px = np.eye(4, dtype=np.float32)
    datastore.save_local_stage_position_zyx_um(
        stage_zyx_um=stage_zyx_um,
        affine_zyx_px=affine_zyx_px,
        tile=0,
        round=0,
    )
    datastore.save_local_wavelengths_um(
        wavelengths_um=(0.488, 0.520), tile=0, round=0
    )
    datastore.save_local_wavelengths_um(wavelengths_um=(0.561, 0.580), tile=0, bit=0)

    corrected = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    datastore.save_local_corrected_image(corrected, tile=0, round=0, psf_idx=0)
    datastore.save_local_corrected_image(corrected, tile=0, bit=0, psf_idx=0)
    datastore.save_local_registered_image(corrected, tile=0, round=0)
    datastore.save_local_registered_image(corrected, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        corrected.astype(np.float32), tile=0, bit=0
    )
    datastore.save_local_feature_predictor_spots(
        pd.DataFrame({"z": [0], "y": [0], "x": [0]}), tile=0, bit=0
    )

    state = dict(datastore.datastore_state)
    state["Corrected"] = True
    state["LocalRegistered"] = True
    datastore.datastore_state = state


def _make_old_v04_dataset(datastore_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    datastore_root.mkdir(parents=True, exist_ok=True)

    state = {
        "Version": 0.4,
        "Initialized": True,
        "Calibrations": True,
        "Corrected": True,
        "LocalRegistered": False,
        "GlobalRegistered": False,
        "Fused": False,
        "SegmentedCells": False,
        "DecodedSpots": False,
        "FilteredSpots": False,
        "RefinedSpots": False,
        "mtxOutput": False,
        "BaysorPath": "",
        "BaysorOptions": "",
        "JuliaThreads": "0",
    }
    qi2labDataStore._save_to_json(state, datastore_root / Path("datastore_state.json"))

    calibrations = datastore_root / Path("calibrations.zarr")
    calibrations.mkdir(parents=True, exist_ok=True)
    calibrations_attrs = {
        "num_rounds": 1,
        "num_tiles": 1,
        "channels_in_data": [0, 1, 2],
        "tile_overlap": 0.1,
        "binning": 1,
        "e_per_ADU": 1.0,
        "na": 1.35,
        "ri": 1.47,
        "exp_order": [[1, 1, 0]],
        "codebook": [["gene_a", 1], ["gene_b", 0]],
        "num_bits": 1,
        "microscope_type": "3D",
        "camera_model": "legacy-camera",
        "voxel_size_zyx_um": [0.5, 0.2, 0.2],
    }
    qi2labDataStore._save_to_json(calibrations_attrs, calibrations / Path(".zattrs"))

    psf_stack = np.arange(3 * 3 * 5 * 5, dtype=np.float32).reshape(3, 3, 5, 5)
    _write_legacy_tensorstore_array(calibrations / Path("psf_data"), psf_stack)

    fiducial_entity = (
        datastore_root / Path("fiducial") / Path("tile0000") / Path("round001.zarr")
    )
    fiducial_entity.mkdir(parents=True, exist_ok=True)
    qi2labDataStore._save_to_json(
        {
            "stage_zyx_um": [1.0, 2.0, 3.0],
            "affine_zyx_px": np.eye(4, dtype=np.float32).tolist(),
            "excitation_um": 0.488,
            "emission_um": 0.520,
            "bit_linker": [1],
            "psf_idx": 0,
        },
        fiducial_entity / Path(".zattrs"),
    )

    readout_entity = (
        datastore_root / Path("readouts") / Path("tile0000") / Path("bit001.zarr")
    )
    readout_entity.mkdir(parents=True, exist_ok=True)
    qi2labDataStore._save_to_json(
        {
            "excitation_um": 0.561,
            "emission_um": 0.580,
            "round_linker": 1,
            "psf_idx": 1,
        },
        readout_entity / Path(".zattrs"),
    )

    fiducial_corrected = np.arange(2 * 6 * 7, dtype=np.uint16).reshape(2, 6, 7)
    readout_corrected = (
        np.arange(2 * 6 * 7, dtype=np.uint16).reshape(2, 6, 7) + 100
    )
    _write_legacy_tensorstore_array(
        fiducial_entity / Path("corrected_data"), fiducial_corrected
    )
    _write_legacy_tensorstore_array(
        readout_entity / Path("corrected_data"), readout_corrected
    )

    return fiducial_corrected, readout_corrected, psf_stack


def test_backward_compat_reads_legacy_tensorstore_array(tmp_path: Path) -> None:
    datastore = _make_calibrated_store(tmp_path / "store")
    datastore.initialize_tile(0)

    legacy_corrected = np.arange(2 * 6 * 7, dtype=np.uint16).reshape(2, 6, 7)
    legacy_path = (
        datastore._fiducial_root_path
        / Path("tile0000")
        / Path("round001.zarr")
        / Path("corrected_data")
    )
    _write_legacy_tensorstore_array(legacy_path, legacy_corrected)

    loaded = datastore.load_local_corrected_image(tile=0, round=0, return_future=False)
    assert isinstance(loaded, np.ndarray)
    np.testing.assert_array_equal(loaded, legacy_corrected)


def test_backward_compat_reads_legacy_linker_aliases(tmp_path: Path) -> None:
    datastore = _make_calibrated_store(tmp_path / "store")
    datastore.initialize_tile(0)

    round_attrs_path = (
        datastore._fiducial_root_path
        / Path("tile0000")
        / Path("round001.zarr")
        / Path(".zattrs")
    )
    round_attrs = datastore._load_from_json(round_attrs_path)
    round_attrs.pop("bit_linker", None)
    round_attrs["bits"] = [0, 7]
    datastore._save_to_json(round_attrs, round_attrs_path)

    bit_attrs_path = (
        datastore._readouts_root_path
        / Path("tile0000")
        / Path("bit001.zarr")
        / Path(".zattrs")
    )
    bit_attrs = datastore._load_from_json(bit_attrs_path)
    bit_attrs.pop("round_linker", None)
    bit_attrs["round"] = 1
    datastore._save_to_json(bit_attrs, bit_attrs_path)

    assert datastore.load_local_bit_linker(tile=0, round=0) == [7]
    assert datastore.load_local_round_linker(tile=0, bit=0) == 1


def test_migration_legacy_zattrs_are_written_to_extra_attributes(
    tmp_path: Path,
) -> None:
    datastore = _make_calibrated_store(tmp_path / "store")
    datastore.initialize_tile(0)

    round_entity = (
        datastore._fiducial_root_path / Path("tile0000") / Path("round001.zarr")
    )
    legacy_attrs = datastore._load_from_json(round_entity / Path(".zattrs"))
    legacy_attrs.update(
        {
            "stage_zyx_um": [1.0, 2.0, 3.0],
            "affine_zyx_px": np.eye(4, dtype=np.float32).tolist(),
            "excitation_um": 0.488,
            "emission_um": 0.520,
            "psf_idx": 0,
        }
    )
    datastore._save_to_json(legacy_attrs, round_entity / Path(".zattrs"))

    corrected = np.arange(3 * 6 * 6, dtype=np.uint16).reshape(3, 6, 6)
    datastore.save_local_corrected_image(
        corrected,
        tile=0,
        round=0,
        gain_correction=True,
        hotpixel_correction=True,
        shading_correction=False,
        psf_idx=2,
    )

    zarr_json = datastore._load_from_json(round_entity / Path("corrected_data/zarr.json"))
    extra_attrs = zarr_json.get("extra_attributes", {})
    assert isinstance(extra_attrs, dict)
    assert extra_attrs["stage_zyx_um"] == [1.0, 2.0, 3.0]
    assert extra_attrs["excitation_um"] == 0.488
    assert extra_attrs["emission_um"] == 0.520
    assert extra_attrs["bit_linker"] == [1, 0]
    assert extra_attrs["psf_idx"] == 2
    assert extra_attrs["gain_correction"] is True
    assert extra_attrs["hotpixel_correction"] is True
    assert extra_attrs["shading_correction"] is False


def test_regression_ragged_psfs_round_trip(tmp_path: Path) -> None:
    datastore_path = tmp_path / "store"
    datastore = _make_calibrated_store(datastore_path, ragged_psfs=True)

    parsed = qi2labDataStore(datastore_path)
    psfs = parsed.channel_psfs
    assert isinstance(psfs, list)
    assert [tuple(psf.shape) for psf in psfs] == [(3, 5, 5), (5, 7, 7), (1, 3, 3)]

    psf_root = datastore._calibrations_zarr_path / Path("psf_data")
    assert (psf_root / Path("psf_000")).exists()
    assert (psf_root / Path("psf_001")).exists()
    assert (psf_root / Path("psf_002")).exists()


def test_regression_local_registered_shape_validation_on_parse(tmp_path: Path) -> None:
    datastore_path = tmp_path / "store"
    datastore = _make_calibrated_store(datastore_path)
    _populate_minimal_local_registered_store(datastore)

    # Baseline parse should succeed.
    qi2labDataStore(datastore_path)

    mismatched_feature_predictor = np.ones((3, 8, 9), dtype=np.float32)
    mismatched_path = (
        datastore._readouts_root_path
        / Path("tile0000")
        / Path("bit001.zarr")
        / Path(f"registered_{datastore.feature_predictor_folder_name}_data")
    )
    _write_legacy_tensorstore_array(mismatched_path, mismatched_feature_predictor)

    with pytest.raises(ValueError, match="shapes differ"):
        qi2labDataStore(datastore_path)


def test_regression_fused_metadata_loads_from_extra_attributes(tmp_path: Path) -> None:
    datastore = _make_calibrated_store(tmp_path / "store")

    fused = np.arange(4 * 6 * 6, dtype=np.uint16).reshape(4, 6, 6)
    affine_zyx_um = np.eye(4, dtype=np.float32)
    origin_zyx_um = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    spacing_zyx_um = np.asarray([0.5, 0.2, 0.2], dtype=np.float32)
    datastore.save_global_fidicual_image(
        fused_image=fused,
        affine_zyx_um=affine_zyx_um,
        origin_zyx_um=origin_zyx_um,
        spacing_zyx_um=spacing_zyx_um,
    )

    fused_path = (
        datastore._fused_root_path
        / Path("fused.zarr")
        / Path(f"fused_{datastore.fiducial_folder_name}_iso_zyx")
    )
    zattrs_path = fused_path / Path(".zattrs")
    if zattrs_path.exists():
        zattrs_path.unlink()

    loaded = datastore.load_global_fidicual_image(return_future=False)
    assert loaded is not None
    loaded_fused, loaded_affine, loaded_origin, loaded_spacing = loaded
    np.testing.assert_array_equal(loaded_fused, fused)
    np.testing.assert_allclose(loaded_affine, affine_zyx_um)
    np.testing.assert_allclose(loaded_origin, origin_zyx_um)
    np.testing.assert_allclose(loaded_spacing, spacing_zyx_um)


def test_new_api_reads_old_v04_dataset(tmp_path: Path) -> None:
    datastore_root = tmp_path / Path("old_v04_store")
    fiducial_corrected, readout_corrected, psf_stack = _make_old_v04_dataset(
        datastore_root
    )

    datastore = qi2labDataStore(datastore_root)

    assert datastore.num_rounds == 1
    assert datastore.num_tiles == 1
    assert datastore.num_bits == 1
    assert datastore.round_ids == ["round001"]
    assert datastore.bit_ids == ["bit001"]
    assert datastore.tile_ids == ["tile0000"]

    loaded_fid = datastore.load_local_corrected_image(
        tile=0, round=0, return_future=False
    )
    loaded_bit = datastore.load_local_corrected_image(
        tile=0, bit=0, return_future=False
    )
    assert isinstance(loaded_fid, np.ndarray)
    assert isinstance(loaded_bit, np.ndarray)
    np.testing.assert_array_equal(loaded_fid, fiducial_corrected)
    np.testing.assert_array_equal(loaded_bit, readout_corrected)

    stage_zyx_um, affine_zyx_px = datastore.load_local_stage_position_zyx_um(
        tile=0, round=0
    )
    np.testing.assert_allclose(stage_zyx_um, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(affine_zyx_px, np.eye(4, dtype=np.float32))

    ex_round, em_round = datastore.load_local_wavelengths_um(tile=0, round=0)
    ex_bit, em_bit = datastore.load_local_wavelengths_um(tile=0, bit=0)
    assert ex_round == pytest.approx(0.488)
    assert em_round == pytest.approx(0.520)
    assert ex_bit == pytest.approx(0.561)
    assert em_bit == pytest.approx(0.580)
    assert datastore.load_local_bit_linker(tile=0, round=0) == [1]
    assert datastore.load_local_round_linker(tile=0, bit=0) == 1

    loaded_psfs = datastore.channel_psfs
    assert isinstance(loaded_psfs, np.ndarray)
    np.testing.assert_array_equal(loaded_psfs, psf_stack)
