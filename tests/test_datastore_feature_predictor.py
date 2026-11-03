import pandas as pd
import pytest


@pytest.mark.unit
def test_feature_predictor_spot_directory_creation_is_idempotent(image_datastore):
    spot_df = pd.DataFrame({"z": [1], "y": [2], "x": [3]})
    for _ in range(2):
        image_datastore.save_local_feature_predictor_spots(
            spot_df, tile="tile0000", bit="bit001"
        )
    pd.testing.assert_frame_equal(
        image_datastore.load_local_feature_predictor_spots(
            tile="tile0000", bit="bit001"
        ),
        spot_df,
    )


@pytest.mark.unit
def test_feature_predictor_spots_preserve_tile_and_bit_on_reopen(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    expected = {}
    for tile in range(2):
        for bit in range(2):
            frame = pd.DataFrame(
                {"z": [tile + 0.5], "y": [bit + 1.5], "x": [tile * 10 + bit]}
            )
            image_datastore.save_local_feature_predictor_spots(frame, tile, bit)
            expected[(tile, bit)] = frame

    root = image_datastore.datastore_path / "ufish" / "localizations"
    assert (root / "tile0000" / "bit001.parquet").is_file()
    assert (root / "tile0001" / "bit002.parquet").is_file()
    reopened = qi2labDataStore(image_datastore.datastore_path)
    for (tile, bit), frame in expected.items():
        tile_id = f"tile{tile:04d}"
        bit_id = f"bit{bit + 1:03d}"
        pd.testing.assert_frame_equal(
            reopened.load_local_feature_predictor_spots(tile_id, bit_id), frame
        )
