from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from merfish3danalysis.cli.qi2lab_microscopes import preprocess
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.integration
@pytest.mark.parametrize("image_datastore", ["one_tile"], indirect=True)
def test_global_fusion_only_creates_fusion_without_existing_transform(image_datastore):
    image_datastore.voxel_size_zyx_um = [1, 1, 1]
    image = np.zeros((3, 10, 12), dtype=np.uint16)
    image[:, 2:-2, 2:-2] = np.arange(144).reshape(3, 6, 8) + 1
    image_datastore.save_local_corrected_image(image, tile=0, round=0)
    image_datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], np.eye(4), tile=0, round=0
    )

    preprocess.local_register_data(
        root_path=image_datastore.datastore_path,
        global_fusion_only=True,
    )

    fused, affine, origin, spacing = image_datastore.load_global_fiducial_image(False)
    projection, _, projection_origin, projection_spacing = (
        image_datastore.load_global_fiducial_projection(False)
    )
    np.testing.assert_array_equal(np.squeeze(fused), image)
    np.testing.assert_array_equal(projection, image.max(axis=0))
    np.testing.assert_array_equal(affine, np.eye(4))
    np.testing.assert_array_equal(origin, [10, 20, 30])
    np.testing.assert_array_equal(spacing, [1, 1, 1])
    np.testing.assert_array_equal(projection_origin, origin)
    np.testing.assert_array_equal(projection_spacing, spacing)


@pytest.mark.integration
def test_global_fusion_only_reuses_stored_tile_transforms(image_datastore):
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [1, 1, 1]
    expected = []
    for tile, (stage, correction, bead, intensity) in enumerate(
        [
            ([10, 20, 30], [0, 1, 2], [2, 6, 4], 10000),
            ([10, 20, 43], [0, 2, 4], [2, 8, 13], 20000),
        ]
    ):
        image = np.zeros((5, 14, 18), dtype=np.uint16)
        z, y, x = bead
        image[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = intensity
        affine = np.eye(4)
        affine[:3, 3] = correction
        datastore.save_local_stage_position_zyx_um(stage, np.eye(4), tile=tile, round=0)
        datastore.save_global_coord_xforms_um(affine, stage, [1, 1, 1], tile=tile)
        datastore.save_local_corrected_image(image, tile=tile, round=0)
        expected.append(
            (affine, stage, np.add(np.add(stage, correction), bead), intensity)
        )

    result = CliRunner().invoke(
        preprocess.app, [str(datastore.datastore_path), "--global-fusion-only"]
    )
    assert result.exit_code == 0, (result.output, result.exception)

    reopened = qi2labDataStore(datastore.datastore_path)
    fused, affine, origin, spacing = reopened.load_global_fiducial_image(False)
    projection, _, projection_origin, projection_spacing = (
        reopened.load_global_fiducial_projection(False)
    )
    fused = np.squeeze(fused)
    np.testing.assert_array_equal(affine, np.eye(4))
    np.testing.assert_array_equal(spacing, [1, 1, 1])
    np.testing.assert_array_equal(projection_origin, origin)
    np.testing.assert_array_equal(projection_spacing, spacing)
    np.testing.assert_array_equal(projection, fused.max(axis=0))
    for tile, (expected_affine, stage, bead_world, intensity) in enumerate(expected):
        stored_affine, stored_origin, stored_spacing = (
            reopened.load_global_coord_xforms_um(tile)
        )
        np.testing.assert_array_equal(stored_affine, expected_affine)
        np.testing.assert_array_equal(stored_origin, stage)
        np.testing.assert_array_equal(stored_spacing, [1, 1, 1])
        bead_index = np.rint((bead_world - origin) / spacing).astype(int)
        assert fused[tuple(bead_index)] == intensity


@pytest.mark.unit
@pytest.mark.parametrize(
    "selected_modes",
    [
        {"global_registration_only": True, "global_fusion_only": True},
        {"global_registration_only": True, "fiducial_registration_only": True},
        {"global_fusion_only": True, "fiducial_registration_only": True},
    ],
)
def test_preprocessing_modes_are_mutually_exclusive(
    selected_modes: dict[str, bool],
) -> None:
    with pytest.raises(typer.BadParameter):
        preprocess.local_register_data(
            root_path=Path("/unused/experiment"),
            **selected_modes,
        )
