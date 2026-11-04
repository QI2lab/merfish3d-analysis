import numpy as np
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


@pytest.mark.integration
def test_spot_table_must_match_current_predictor_image(image_datastore):
    from merfish3danalysis.DataRegistration import DataRegistration

    datastore = image_datastore
    image = np.ones((2, 8, 8), dtype=np.uint16)
    spots = pd.DataFrame({"z": [0], "y": [2], "x": [3]})
    datastore.save_local_corrected_image(image, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        image, tile=0, bit=0, source_image="corrected_data"
    )
    datastore.save_local_feature_predictor_spots(spots, tile=0, bit=0)
    registration = DataRegistration(datastore, decon_readout=False, verbose=0)
    assert registration._has_valid_feature_predictor_outputs("tile0000", "bit001")

    datastore.save_local_feature_predictor_image(
        image * 2, tile=0, bit=0, source_image="corrected_data"
    )
    assert not registration._has_valid_feature_predictor_outputs("tile0000", "bit001")
    with pytest.raises(ValueError, match=r"predictor.*changed"):
        datastore.load_local_feature_predictor_spots(0, 0)

    new_spots = pd.DataFrame({"z": [0], "y": [4], "x": [5]})
    datastore.save_local_feature_predictor_spots(new_spots, tile=0, bit=0)
    assert registration._has_valid_feature_predictor_outputs("tile0000", "bit001")
    pd.testing.assert_frame_equal(
        datastore.load_local_feature_predictor_spots(0, 0), new_spots
    )


@pytest.mark.integration
def test_failed_spot_write_preserves_previous_table_and_requires_resume(
    image_datastore,
):
    import os
    import subprocess
    import sys

    from merfish3danalysis.DataRegistration import DataRegistration

    if os.name != "posix":
        pytest.skip("File size limits require POSIX")
    datastore = image_datastore
    image = np.ones((2, 8, 8), dtype=np.uint16)
    spots = pd.DataFrame({"z": [0], "y": [2], "x": [3]})
    datastore.save_local_corrected_image(image, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        image, tile=0, bit=0, source_image="corrected_data"
    )
    datastore.save_local_feature_predictor_spots(spots, tile=0, bit=0)
    path = datastore.local_feature_predictor_spots_path("tile0000", "bit001")
    original = path.read_bytes()

    datastore.save_local_feature_predictor_image(
        image * 2, tile=0, bit=0, source_image="corrected_data"
    )
    script = """
import errno
import resource
import sys
import pandas as pd
from merfish3danalysis.qi2labDataStore import qi2labDataStore

datastore = qi2labDataStore(sys.argv[1])
_, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (0, hard_limit))
try:
    datastore.save_local_feature_predictor_spots(
        pd.DataFrame({'z': [0], 'y': [4], 'x': [5]}), tile=0, bit=0
    )
except OSError as exc:
    assert exc.errno == errno.EFBIG
else:
    raise AssertionError('expected a file-size limit error')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(datastore.datastore_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == original
    registration = DataRegistration(datastore, decon_readout=False, verbose=0)
    assert not registration._has_valid_feature_predictor_outputs("tile0000", "bit001")
    with pytest.raises(ValueError, match=r"predictor.*changed"):
        datastore.load_local_feature_predictor_spots(0, 0)

    new_spots = pd.DataFrame({"z": [0], "y": [4], "x": [5]})
    datastore.save_local_feature_predictor_spots(new_spots, tile=0, bit=0)
    assert registration._has_valid_feature_predictor_outputs("tile0000", "bit001")
    pd.testing.assert_frame_equal(
        datastore.load_local_feature_predictor_spots(0, 0), new_spots
    )


@pytest.mark.integration
def test_untracked_spot_table_remains_readable_but_is_not_complete(image_datastore):
    from merfish3danalysis.DataRegistration import DataRegistration

    datastore = image_datastore
    image = np.ones((2, 8, 8), dtype=np.uint16)
    spots = pd.DataFrame({"z": [0], "y": [2], "x": [3]})
    datastore.save_local_corrected_image(image, tile=0, bit=0)
    datastore.save_local_feature_predictor_spots(spots, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        image, tile=0, bit=0, source_image="corrected_data"
    )
    pd.testing.assert_frame_equal(
        datastore.load_local_feature_predictor_spots(0, 0), spots
    )
    registration = DataRegistration(datastore, decon_readout=False, verbose=0)
    assert not registration._has_valid_feature_predictor_outputs("tile0000", "bit001")
