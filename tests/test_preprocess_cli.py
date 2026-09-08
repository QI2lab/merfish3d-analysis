from pathlib import Path
from unittest.mock import Mock, sentinel

import pytest
import typer

import merfish3danalysis.DataRegistration as data_registration_module
from merfish3danalysis.cli.qi2lab_microscopes import preprocess


def test_global_fusion_only_uses_stored_transform_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fuse_global_registered = Mock()

    class _FakeDataRegistration:
        def __init__(self, **_kwargs) -> None:
            pass

        def fuse_global_registered(self, *, create_max_proj_tiff: bool) -> None:
            fuse_global_registered(create_max_proj_tiff=create_max_proj_tiff)

        def global_register(self, *, create_max_proj_tiff: bool) -> None:
            raise AssertionError("global registration must not run")

        def register_all_tiles(self, **_kwargs) -> None:
            raise AssertionError("local registration must not run")

    monkeypatch.setattr(
        data_registration_module,
        "DataRegistration",
        _FakeDataRegistration,
    )
    monkeypatch.setattr(
        preprocess,
        "qi2lab_datastore_path",
        lambda _root_path: Path("/unused/qi2labdatastore"),
    )
    monkeypatch.setattr(
        preprocess,
        "qi2labDataStore",
        lambda _datastore_path: sentinel.datastore,
    )

    preprocess.local_register_data(
        root_path=Path("/unused/experiment"),
        global_fusion_only=True,
    )

    fuse_global_registered.assert_called_once_with(create_max_proj_tiff=True)


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


def test_no_decon_disables_readout_deconvolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_kwargs = {}
    register_all_tiles = Mock()

    class _FakeDataRegistration:
        def __init__(self, **kwargs) -> None:
            constructor_kwargs.update(kwargs)

        def register_all_tiles(self, **kwargs) -> None:
            register_all_tiles(**kwargs)

    monkeypatch.setattr(
        data_registration_module,
        "DataRegistration",
        _FakeDataRegistration,
    )
    monkeypatch.setattr(
        preprocess,
        "qi2lab_datastore_path",
        lambda _root_path: Path("/unused/qi2labdatastore"),
    )
    datastore = Mock()
    datastore.datastore_state = {}
    monkeypatch.setattr(
        preprocess,
        "qi2labDataStore",
        lambda _datastore_path: datastore,
    )

    preprocess.local_register_data(
        root_path=Path("/unused/experiment"),
        decon=False,
    )

    assert constructor_kwargs["decon_fiducial"] is True
    assert constructor_kwargs["decon_readout"] is False
    register_all_tiles.assert_called_once_with(process_readouts=True)


def test_fiducial_registration_only_skips_readout_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_all_tiles = Mock()

    class _FakeDataRegistration:
        def __init__(self, **_kwargs) -> None:
            pass

        def register_all_tiles(self, **kwargs) -> None:
            register_all_tiles(**kwargs)

    monkeypatch.setattr(
        data_registration_module,
        "DataRegistration",
        _FakeDataRegistration,
    )
    monkeypatch.setattr(
        preprocess,
        "qi2lab_datastore_path",
        lambda _root_path: Path("/unused/qi2labdatastore"),
    )
    datastore = Mock()
    datastore.datastore_state = {}
    monkeypatch.setattr(
        preprocess,
        "qi2labDataStore",
        lambda _datastore_path: datastore,
    )

    preprocess.local_register_data(
        root_path=Path("/unused/experiment"),
        fiducial_registration_only=True,
    )

    register_all_tiles.assert_called_once_with(process_readouts=False)
