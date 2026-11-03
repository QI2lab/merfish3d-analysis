import numpy as np
import pytest

from merfish3danalysis.DataRegistration import DataRegistration


@pytest.mark.unit
def test_global_transforms_available_requires_every_tile(image_datastore) -> None:
    registration = DataRegistration(image_datastore, verbose=0)
    identity = np.eye(4, dtype=np.float32)
    assert not registration._global_transforms_available()
    for tile in image_datastore.tile_ids:
        image_datastore.save_global_coord_xforms_um(
            identity, [0, 0, 0], image_datastore.voxel_size_zyx_um, tile=tile
        )
        assert registration._global_transforms_available() == (
            tile == image_datastore.tile_ids[-1]
        )
