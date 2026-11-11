import numpy as np
import pytest

from merfish3danalysis.utils.decode_warping import warp_bit_image_to_reference


@pytest.mark.unit
@pytest.mark.gpu
def test_decode_samples_persisted_round_chromatic_and_deformable_transforms(
    image_datastore,
):
    """A readout is sampled once using the stored reference-to-native chain."""
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    datastore_path = datastore.datastore_path
    datastore.voxel_size_zyx_um = [2.0, 1.0, 0.5]

    z, y, x = np.indices((8, 10, 12), dtype=np.float32)
    image = 100 * z + 10 * y + x
    datastore.save_local_corrected_image(image, tile=0, bit=1)
    round_transform = np.array(
        [[1, 0, 0, 2], [0, 0, -1, 5], [0, 1, 0, 1], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    chromatic_transform = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    datastore.save_local_round_transform_zyx_um(round_transform, tile=0, round=1)
    datastore.save_chromatic_affine_transforms_zyx_um(
        {
            "estimator": "decoded_rna_on_bit_weighted_centroid_z_translation_yx_affine_graph",
            "channels": {
                "readout": {
                    "wavelength_um": 0.67,
                    "affine_zyx_um": chromatic_transform,
                    "status": "ok",
                }
            },
        }
    )

    # The first field sample is at ZYX=(1, 1, 2), with stride (2, 2, 4).
    # At reference pixel (1, 2, 4) the displacement is ZYX=(0.5, -1, 2).
    flow = np.empty((3, 2, 3, 3), dtype=np.float32)
    flow[0] = [1, 3, 5]
    flow[1] = -1
    flow[2] = 0.5
    datastore.save_local_sofima_flow_field(
        flow,
        tile=0,
        round=1,
        reference_round=0,
        map_stride_zyx_px=(2, 2, 4),
        map_box_start_xyz_px=(2, 1, 1),
        map_box_size_xyz_px=(12, 10, 8),
        reference_shape_zyx_px=image.shape,
        moving_shape_zyx_px=image.shape,
        sofima_status="ok",
    )

    datastore = qi2labDataStore(datastore_path, validate=False)
    native = datastore.load_local_corrected_image(tile=0, bit=1, return_future=False)
    observed = warp_bit_image_to_reference(
        native,
        datastore=datastore,
        tile=0,
        bit_id="bit002",
        emission_wavelength_um=0.67,
    )

    # D, spacing, round affine, inverse chromatic affine, inverse spacing map
    # reference pixel (1, 2, 4) to native (2.5, 1.5, 4). The image is a
    # linear coordinate ramp, giving an independent trilinear result of 269.
    assert observed[1, 2, 4] == pytest.approx(269.0, abs=1e-4)
    datastore.save_chromatic_affine_transforms_zyx_um(
        {
            "estimator": "bead_calibration",
            "channels": {
                "readout": {"wavelength_um": 0.67, "affine_zyx_um": chromatic_transform}
            },
        }
    )
    bead_warp = warp_bit_image_to_reference(
        native,
        datastore=datastore,
        tile=0,
        bit_id="bit002",
        emission_wavelength_um=0.67,
    )
    # With no applicable optimizer estimate, chromatic correction is identity.
    assert bead_warp[1, 2, 4] == pytest.approx(274.0, abs=1e-4)
