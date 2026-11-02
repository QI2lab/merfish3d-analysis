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
