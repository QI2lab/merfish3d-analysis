import sys
import types

import numpy as np

from merfish3danalysis.cli.qi2lab_microscopes import fuseall


class _FakeDatastore:
    def __init__(self) -> None:
        self.decon = np.array(
            [
                [[10, 20], [30, 40]],
                [[50, 60], [70, 80]],
            ],
            dtype=np.uint16,
        )
        self.calls = []

    def load_local_registered_image(self, *, tile, bit, return_future):
        self.calls.append(("decon", tile, bit, return_future))
        return self.decon

    def load_local_feature_predictor_image(self, *, tile, bit, return_future):
        raise AssertionError("fuseall should warp raw readout data without predictors")

    def load_local_wavelengths_um(self, *, tile, bit):
        self.calls.append(("wavelength", tile, bit))
        return 0.561, 0.670


def test_load_warped_readout_image_for_fusion_uses_local_registration_chain(
    monkeypatch,
) -> None:
    datastore = _FakeDatastore()
    observed = {}

    def fake_warp(
        image,
        *,
        datastore,
        tile,
        bit_id,
        emission_wavelength_um,
        gpu_id,
    ):
        observed["image"] = image.copy()
        observed["datastore"] = datastore
        observed["tile"] = tile
        observed["bit_id"] = bit_id
        observed["emission_wavelength_um"] = emission_wavelength_um
        observed["gpu_id"] = gpu_id
        return image + 1

    monkeypatch.setattr(fuseall, "warp_bit_image_to_reference", fake_warp)

    warped = fuseall._load_warped_readout_image_for_fusion(
        datastore,
        tile_id="tile0000",
        bit_id="bit002",
        gpu_id=3,
    )

    expected_readout = datastore.decon.astype(np.float32)
    np.testing.assert_allclose(observed["image"], expected_readout)
    assert observed["datastore"] is datastore
    assert observed["tile"] == "tile0000"
    assert observed["bit_id"] == "bit002"
    assert observed["emission_wavelength_um"] == 0.670
    assert observed["gpu_id"] == 3
    np.testing.assert_array_equal(warped, (expected_readout + 1).astype(np.uint16))
    assert datastore.calls == [
        ("decon", "tile0000", "bit002", False),
        ("wavelength", "tile0000", "bit002"),
    ]


def test_attach_stored_global_transform_uses_saved_global_registration() -> None:
    class DatastoreWithGlobalTransform:
        def load_global_coord_xforms_um(self, *, tile):
            assert tile == "tile0000"
            return np.eye(4, dtype=np.float32) * 2, np.zeros(3), np.ones(3)

    class MsiUtils:
        def __init__(self) -> None:
            self.call = None

        def set_affine_transform(self, msim, affine, *, transform_key):
            self.call = (msim, affine, transform_key)

    msim = object()
    msi_utils = MsiUtils()

    fuseall._attach_stored_global_transform(
        msim,
        datastore=DatastoreWithGlobalTransform(),
        tile_id="tile0000",
        msi_utils=msi_utils,
        transform_key="global_registered",
    )

    called_msim, affine, transform_key = msi_utils.call
    assert called_msim is msim
    np.testing.assert_array_equal(
        affine,
        (np.eye(4, dtype=np.float32) * 2)[None, ...],
    )
    assert transform_key == "global_registered"


def test_channel_output_name_uses_fiducial_and_zero_padded_bits() -> None:
    assert fuseall._channel_output_name("fiducial") == "fiducial"
    assert fuseall._channel_output_name("bit1") == "bit001"
    assert fuseall._channel_output_name("bit001") == "bit001"
    assert fuseall._channel_output_name("12") == "bit012"


def test_write_fused_ome_tiff_reads_completed_zarr_and_writes_tiff(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeSim:
        data = np.array([[[[0, 2**16], [2, 3]]]], dtype=np.float32)

    class FakeNgffUtils:
        def __init__(self) -> None:
            self.call = None

        def read_sim_from_ome_zarr(
            self,
            ome_zarr_path,
            *,
            resolution_level,
            transform_key,
            use_dask,
        ):
            self.call = (
                ome_zarr_path,
                resolution_level,
                transform_key,
                use_dask,
            )
            return FakeSim()

    written = {}

    class FakeTiffWriter:
        def __init__(self, path, *, bigtiff):
            written["path"] = path
            written["bigtiff"] = bigtiff

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def write(self, image, **kwargs):
            written["image"] = image
            written["kwargs"] = kwargs

    fake_tifffile = types.ModuleType("tifffile")
    fake_tifffile.TiffWriter = FakeTiffWriter
    monkeypatch.setitem(sys.modules, "tifffile", fake_tifffile)

    ngff_utils = FakeNgffUtils()
    ome_zarr_path = tmp_path / "bit001.ome.zarr"
    ome_tiff_path = tmp_path / "bit001.ome.tiff"

    fuseall._write_fused_ome_tiff(
        ome_zarr_path=ome_zarr_path,
        ome_tiff_path=ome_tiff_path,
        ngff_utils=ngff_utils,
        spacing_zyx_um=(0.5, 0.2, 0.2),
        transform_key="global_registered",
    )

    assert ngff_utils.call == (ome_zarr_path, 0, "global_registered", False)
    assert written["path"] == ome_tiff_path
    assert written["bigtiff"] is True
    np.testing.assert_array_equal(
        written["image"],
        np.array([[[0, 65535], [2, 3]]], dtype=np.uint16),
    )
    assert written["kwargs"]["metadata"]["axes"] == "ZYX"
    assert written["kwargs"]["metadata"]["PhysicalSizeX"] == 0.2
    assert written["kwargs"]["metadata"]["PhysicalSizeY"] == 0.2
    assert written["kwargs"]["metadata"]["PhysicalSizeZ"] == 0.5
