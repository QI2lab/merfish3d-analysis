from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from merfish3danalysis.qi2labDataStore import qi2labDataStore


def test_feature_predictor_spot_directory_creation_is_idempotent(
    tmp_path: Path,
) -> None:
    datastore = qi2labDataStore.__new__(qi2labDataStore)
    datastore._num_tiles = 1
    datastore._tile_ids = ["tile0000"]
    datastore._bit_ids = ["bit001"]
    datastore._feature_predictor_localizations_root_path = tmp_path
    datastore._save_to_parquet = Mock()
    spot_df = pd.DataFrame({"z": [1], "y": [2], "x": [3]})

    with patch.object(Path, "mkdir") as mkdir:
        datastore.save_local_feature_predictor_spots(
            spot_df,
            tile="tile0000",
            bit="bit001",
        )

    mkdir.assert_called_once_with(parents=True, exist_ok=True)
    datastore._save_to_parquet.assert_called_once_with(
        spot_df,
        tmp_path / "tile0000" / "bit001.parquet",
    )
