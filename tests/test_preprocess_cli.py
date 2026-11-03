from pathlib import Path

import numpy as np
import pytest
import typer

from merfish3danalysis.cli.qi2lab_microscopes import preprocess


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
