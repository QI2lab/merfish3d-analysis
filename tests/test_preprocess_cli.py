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

        def register_all_tiles(self) -> None:
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


def test_global_only_modes_are_mutually_exclusive() -> None:
    with pytest.raises(typer.BadParameter):
        preprocess.local_register_data(
            root_path=Path("/unused/experiment"),
            global_registration_only=True,
            global_fusion_only=True,
        )
