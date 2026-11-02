from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from merfish3danalysis.cli.qi2lab_microscopes import preprocess
from merfish3danalysis.DataRegistration import DataRegistration


@pytest.mark.unit
def test_global_fusion_only_uses_stored_transform_fusion(image_datastore):
    with patch.object(
        DataRegistration, "fuse_global_registered", autospec=True
    ) as fuse:
        preprocess.local_register_data(
            root_path=image_datastore.datastore_path,
            global_fusion_only=True,
        )

    fuse.assert_called_once()
    registration = fuse.call_args.args[0]
    assert registration.datastore.datastore_path == image_datastore.datastore_path
    assert fuse.call_args.kwargs == {"create_max_proj_tiff": True}


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


@pytest.mark.unit
def test_no_decon_disables_readout_deconvolution(image_datastore):
    with patch.object(
        DataRegistration, "register_all_tiles", autospec=True
    ) as register:
        preprocess.local_register_data(
            root_path=image_datastore.datastore_path,
            decon=False,
        )

    register.assert_called_once()
    registration = register.call_args.args[0]
    assert registration.datastore.datastore_path == image_datastore.datastore_path
    assert registration._decon_fiducial is True
    assert registration._decon_readout is False
    assert register.call_args.kwargs == {"process_readouts": True}


@pytest.mark.unit
def test_fiducial_registration_only_skips_readout_processing(image_datastore):
    with patch.object(
        DataRegistration, "register_all_tiles", autospec=True
    ) as register:
        preprocess.local_register_data(
            root_path=image_datastore.datastore_path,
            fiducial_registration_only=True,
        )

    register.assert_called_once()
    registration = register.call_args.args[0]
    assert registration.datastore.datastore_path == image_datastore.datastore_path
    assert register.call_args.kwargs == {"process_readouts": False}
