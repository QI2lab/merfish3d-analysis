import sys
import types
from typing import ClassVar

import numpy as np

from merfish3danalysis.utils.decode_warping import warp_bit_image_to_reference


class _NoChromaticDatastore:
    round_ids: ClassVar[list[str]] = ["round001", "round002"]
    voxel_size_zyx_um = (0.5, 0.2, 0.2)

    def __init__(self) -> None:
        self.round_transform = np.array(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def load_local_round_linker(self, *, tile, bit):
        return 2

    def load_local_round_transform_zyx_um(self, *, tile, round):
        assert round == "round002"
        return self.round_transform

    def load_chromatic_affine_transform_zyx_um(self, *, wavelength_um):
        return None

    def load_local_sofima_flow_field(self, *, tile, round, return_future):
        return None


def test_warp_bit_image_to_reference_uses_identity_without_chromatic_affine(
    monkeypatch,
) -> None:
    datastore = _NoChromaticDatastore()
    image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    observed = {}

    def fake_warp_array_to_reference_gpu(
        image,
        *,
        transform_zyx_um,
        spacing_zyx_um,
        reference_shape,
        gpu_id,
    ):
        observed["image"] = image
        observed["transform_zyx_um"] = transform_zyx_um
        observed["spacing_zyx_um"] = spacing_zyx_um
        observed["reference_shape"] = reference_shape
        observed["gpu_id"] = gpu_id
        return image + 10

    fake_multiview_registration = types.ModuleType(
        "merfish3danalysis.utils.multiview_registration"
    )
    fake_multiview_registration.warp_array_to_reference_gpu = (
        fake_warp_array_to_reference_gpu
    )
    monkeypatch.setitem(
        sys.modules,
        "merfish3danalysis.utils.multiview_registration",
        fake_multiview_registration,
    )

    warped = warp_bit_image_to_reference(
        image,
        datastore=datastore,
        tile="tile0000",
        bit_id="bit002",
        emission_wavelength_um=0.670,
        gpu_id=4,
    )

    np.testing.assert_array_equal(observed["image"], image)
    np.testing.assert_allclose(observed["transform_zyx_um"], datastore.round_transform)
    assert observed["spacing_zyx_um"] == datastore.voxel_size_zyx_um
    assert observed["reference_shape"] == image.shape
    assert observed["gpu_id"] == 4
    np.testing.assert_array_equal(warped, image + 10)
