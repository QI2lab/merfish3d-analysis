from types import SimpleNamespace

import numpy as np
import pandas as pd

from merfish3danalysis.cli.qi2lab_microscopes.pixeldecode import (
    QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD,
    _default_qi2lab_magnitude_threshold,
    _default_qi2lab_minimum_pixels,
)
from merfish3danalysis.PixelDecoder import PixelDecoder
from merfish3danalysis.qi2labDataStore import qi2labDataStore


def test_datastore_decoded_paths_use_optional_run_subfolders(tmp_path) -> None:
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore", validate=False)
    datastore.num_tiles = 1
    decoded = pd.DataFrame({"gene_id": ["GeneA"], "global_z": [1.0]})

    datastore.save_local_decoded_spots(decoded, tile=0)
    datastore.save_local_decoded_spots(
        decoded,
        tile=0,
        decode_run_key="zstride_03_2d",
    )
    datastore.save_global_filtered_decoded_spots(decoded)
    datastore.save_global_filtered_decoded_spots(
        decoded,
        decode_run_key="zstride_03_2d",
    )

    root = tmp_path / "qi2labdatastore"
    assert (root / "decoded" / "tile0000_decoded_features.parquet").exists()
    assert (
        root / "decoded" / "zstride_03_2d" / "tile0000_decoded_features.parquet"
    ).exists()
    assert (
        root / "all_tiles_filtered_decoded_features" / "decoded_features.parquet"
    ).exists()
    assert (
        root
        / "all_tiles_filtered_decoded_features"
        / "zstride_03_2d"
        / "decoded_features.parquet"
    ).exists()

    default_loaded = datastore.load_local_decoded_spots(tile=0)
    run_loaded = datastore.load_local_decoded_spots(
        tile=0,
        decode_run_key="zstride_03_2d",
    )
    assert default_loaded.equals(decoded)
    assert run_loaded.equals(decoded)


def test_run_normalization_vectors_do_not_overwrite_defaults(tmp_path) -> None:
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore", validate=False)
    datastore.global_normalization_vector = [1.0, 2.0]
    datastore.global_background_vector = [0.1, 0.2]

    datastore.save_decode_normalization_vectors(
        "zstride_03_2d",
        "global",
        normalization_vector=[3.0, 4.0],
        background_vector=[0.3, 0.4],
        zstride_level=3,
        decode_mode="2d",
    )

    default_norm, default_bkd = datastore.load_decode_normalization_vectors(
        None,
        "global",
    )
    run_norm, run_bkd = datastore.load_decode_normalization_vectors(
        "zstride_03_2d",
        "global",
    )

    np.testing.assert_allclose(default_norm, [1.0, 2.0])
    np.testing.assert_allclose(default_bkd, [0.1, 0.2])
    np.testing.assert_allclose(run_norm, [3.0, 4.0])
    np.testing.assert_allclose(run_bkd, [0.3, 0.4])


def test_pixel_decoder_maps_strided_decoded_z_to_source_z() -> None:
    decoder = object.__new__(PixelDecoder)
    decoder._z_range = [5, None]
    decoder._zstride = 3

    decoded_z = pd.Series([0.0, 1.5, 2.0])

    mapped = decoder._decoded_z_to_source_z(decoded_z)

    np.testing.assert_allclose(mapped.to_numpy(), [5.0, 9.5, 11.0])


def test_decode_mode_controls_cli_defaults() -> None:
    datastore = SimpleNamespace(
        microscope_type="3D", voxel_size_zyx_um=[0.945, 0.1, 0.1]
    )

    assert _default_qi2lab_minimum_pixels(datastore, decode_mode="auto") == 28
    assert _default_qi2lab_minimum_pixels(datastore, decode_mode="2d") == 7
    assert (
        _default_qi2lab_magnitude_threshold(datastore, decode_mode="3d")
        == QI2LAB_3D_DEFAULT_MAGNITUDE_THRESHOLD
    )
    assert _default_qi2lab_magnitude_threshold(datastore, decode_mode="2d")[0] == 0.7
