import runpy
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from merfish3danalysis.qi2labDataStore import qi2labDataStore

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/zhuang_lab/02_register_and_deconvolve.py"
)


@pytest.mark.integration
@pytest.mark.parametrize("image_datastore", ["one_tile"], indirect=True)
@pytest.mark.parametrize("datastore_argument", [False, True])
def test_global_registration_cli_creates_projection(
    image_datastore, datastore_argument
):
    datastore = image_datastore
    datastore_path = datastore.datastore_path
    datastore.voxel_size_zyx_um = [1, 1, 1]
    image = np.zeros((3, 12, 16), dtype=np.uint16)
    image[:, 2:-2, 2:-2] = np.arange(288).reshape(3, 8, 12) + 1
    datastore.save_local_corrected_image(image, tile=0, round=0)
    datastore.save_local_stage_position_zyx_um([10, 20, 30], np.eye(4), tile=0, round=0)
    example = runpy.run_path(str(EXAMPLE_PATH))
    result = CliRunner().invoke(
        example["app"],
        [
            str(datastore_path if datastore_argument else datastore_path.parent),
            "--global-only",
        ],
    )
    assert result.exit_code == 0, result.exception
    reopened = qi2labDataStore(datastore_path)
    fused, affine, origin, spacing = reopened.load_global_fiducial_image(False)
    projection, projection_affine, projection_origin, projection_spacing = (
        reopened.load_global_fiducial_projection(False)
    )
    np.testing.assert_array_equal(np.squeeze(fused), image)
    np.testing.assert_array_equal(projection, image.max(axis=0))
    np.testing.assert_array_equal(affine, np.eye(4))
    np.testing.assert_array_equal(origin, [10, 20, 30])
    np.testing.assert_array_equal(spacing, [1, 1, 1])
    np.testing.assert_array_equal(projection_affine, affine)
    np.testing.assert_array_equal(projection_origin, origin)
    np.testing.assert_array_equal(projection_spacing, spacing)
    assert reopened.datastore_state["GlobalRegistered"]
    assert reopened.datastore_state["Fused"]


@pytest.mark.unit
@pytest.mark.parametrize("function", ["local_register_data", "global_register_data"])
def test_registration_rejects_missing_datastore_before_creating_one(tmp_path, function):
    example = runpy.run_path(str(EXAMPLE_PATH))
    with pytest.raises(FileNotFoundError, match="No qi2lab datastore"):
        example[function](tmp_path)
    assert not (tmp_path / "qi2labdatastore").exists()
