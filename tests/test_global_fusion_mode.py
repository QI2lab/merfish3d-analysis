from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from merfish3danalysis.DataRegistration import DataRegistration


def _registration_with_transforms(
    transforms: dict[str, np.ndarray | None],
) -> DataRegistration:
    registration = DataRegistration.__new__(DataRegistration)
    registration._tile_ids = list(transforms)
    registration._verbose = 0
    registration._datastore = SimpleNamespace(
        load_global_coord_xforms_um=lambda *, tile: (
            transforms[tile],
            None,
            None,
        )
    )
    return registration


def test_global_transforms_available_requires_every_tile() -> None:
    identity = np.eye(4, dtype=np.float32)
    complete = _registration_with_transforms(
        {"tile0000": identity, "tile0001": identity}
    )
    partial = _registration_with_transforms(
        {"tile0000": identity, "tile0001": None}
    )

    assert complete._global_transforms_available()
    assert not partial._global_transforms_available()


def test_global_fusion_registers_when_stored_transforms_are_missing() -> None:
    registration = _registration_with_transforms({"tile0000": None})
    registration.global_register = Mock()

    registration.fuse_global_registered(create_max_proj_tiff=True)

    registration.global_register.assert_called_once_with(create_max_proj_tiff=True)
