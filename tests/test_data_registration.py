from queue import SimpleQueue
from unittest.mock import Mock

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import DataRegistration


@pytest.fixture
def registration():
    datastore = Mock(
        tile_ids=["tile0000", "tile0001"],
        num_tiles=2,
        round_ids=["round001", "round002"],
        bit_ids=["bit001"],
        channel_psfs=[],
        voxel_size_zyx_um=[0.5, 0.2, 0.3],
    )
    return DataRegistration(datastore=datastore, verbose=0)


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 1, "tile0000", "tile0001"])
def test_tile_selection_uses_datastore_identifiers(registration, value):
    registration.tile_id = value
    assert registration.tile_id == (
        registration.datastore.tile_ids[value] if isinstance(value, int) else value
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [-1, 2, 3, "missing"])
def test_invalid_tile_selection_preserves_previous_tile(registration, value):
    registration.tile_id = "tile0001"
    registration.tile_id = value
    assert registration.tile_id == "tile0001"


@pytest.mark.unit
@pytest.mark.parametrize("global_registration", [False, True])
@pytest.mark.parametrize("process_readouts", [False, True])
def test_complete_local_tiles_still_run_requested_global_registration(
    registration, global_registration, process_readouts
):
    registration._global_registration = global_registration
    registration._is_tile_complete = Mock(return_value=True)
    registration._generate_registrations = Mock()
    registration._apply_registration_to_bits = Mock()
    registration.global_register = Mock()

    registration.register_all_tiles(process_readouts=process_readouts)

    registration._generate_registrations.assert_not_called()
    registration._apply_registration_to_bits.assert_not_called()
    assert registration.global_register.call_count == int(global_registration)


@pytest.mark.unit
def test_failed_local_registration_prevents_readouts_and_global_success(registration):
    registration._overwrite_outputs = True
    registration._global_registration = True
    registration._generate_registrations = Mock(side_effect=OSError("write failed"))
    registration._apply_registration_to_bits = Mock()
    registration.global_register = Mock()

    with pytest.raises(OSError, match="write failed"):
        registration.register_all_tiles()

    registration._apply_registration_to_bits.assert_not_called()
    registration.global_register.assert_not_called()


@pytest.mark.integration
def test_fiducial_worker_saves_physical_transform_that_restores_generated_image(
    registration, monkeypatch
):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")

    from scipy.ndimage import gaussian_filter

    import merfish3danalysis.DataRegistration as registration_module
    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_gpu,
    )

    fixed = np.zeros((31, 64, 64), dtype=np.float32)
    fixed[8, 18, 20] = 1
    fixed[16, 42, 35] = 2
    fixed[23, 30, 50] = 3
    fixed = gaussian_filter(fixed, sigma=(1, 2, 2))
    # Native round 2 is displaced +2 Z, +3 Y, -4 X pixels.
    moving = np.zeros_like(fixed)
    moving[2:, 3:, :-4] = fixed[:-2, :-3, 4:]
    registration.tile_id = 0
    monkeypatch.setattr(
        registration_module,
        "_load_deconvolve_fiducial_round",
        Mock(return_value=moving),
    )
    # Keep the test process's device visibility unchanged; execute real CUDA math.
    monkeypatch.setattr(
        registration_module, "_restrict_worker_to_assigned_gpu", Mock(return_value=0)
    )
    results = SimpleQueue()

    registration_module._process_fiducial_rounds_on_gpu(
        registration, ["round002"], fixed, 0, results
    )

    saved = registration.datastore.save_local_round_transform_zyx_um.call_args.kwargs
    transform = saved["transform_zyx_um"]
    assert saved["tile"] == "tile0000"
    assert saved["round"] == "round002"
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), rtol=0, atol=1e-7)
    # Pull mapping into the displaced native image, in micrometers.
    np.testing.assert_allclose(transform[:3, 3], [1.0, 0.6, -1.2], rtol=0, atol=0.05)
    np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
    restored = warp_array_to_reference_gpu(
        moving,
        transform_zyx_um=transform,
        spacing_zyx_um=registration.datastore.voxel_size_zyx_um,
        reference_shape=fixed.shape,
    )
    # Ignore boundary samples lost in the explicitly translated native image.
    np.testing.assert_allclose(
        restored[4:-4, 6:-6, 8:-8], fixed[4:-4, 6:-6, 8:-8], atol=1e-4
    )
    assert results.get_nowait() == ("result", "round002", 2048)
    assert results.empty()


@pytest.mark.unit
@pytest.mark.parametrize(
    "method", ["register_one_tile", "apply_registration_to_one_tile"]
)
@pytest.mark.parametrize("tile_id", [-1, 2, "missing"])
def test_invalid_tile_request_never_processes_previously_selected_tile(
    registration, method, tile_id
):
    registration.tile_id = "tile0000"
    registration._generate_registrations = Mock()
    registration._apply_registration_to_bits = Mock()

    getattr(registration, method)(tile_id)

    registration._generate_registrations.assert_not_called()
    registration._apply_registration_to_bits.assert_not_called()


@pytest.mark.integration
def test_readout_worker_preserves_native_pixels_and_clips_spot_rois_at_edges(
    registration, monkeypatch
):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")

    import pandas as pd
    import ufish.api

    import merfish3danalysis.DataRegistration as registration_module

    z, y, x = np.indices((9, 9, 9))
    native = (100 * z + 10 * y + x + 1).astype(np.uint16)
    probabilities = np.full(native.shape, 0.25, dtype=np.float32)
    spots = pd.DataFrame(
        {"axis-0": [4, 0, 8], "axis-1": [4, 0, 8], "axis-2": [4, 0, 8]}
    )
    predictor = Mock()
    predictor.predict.return_value = (spots, probabilities)
    monkeypatch.setattr(ufish.api, "UFish", Mock(return_value=predictor))
    monkeypatch.setattr(registration_module, "load_ufish_model", Mock())
    monkeypatch.setattr(
        registration_module, "_restrict_worker_to_assigned_gpu", Mock(return_value=0)
    )
    registration.tile_id = 0
    registration._has_valid_deconvolved_readout_image = Mock(return_value=False)
    registration._has_valid_feature_predictor_outputs = Mock(return_value=False)
    registration.datastore.load_local_round_linker.return_value = 1
    registration.datastore.load_local_wavelengths_um.return_value = (0.65, 0.67)
    registration.datastore.load_local_corrected_image.return_value = native

    registration_module._apply_bits_on_gpu(registration, ["bit001"], 0)

    np.testing.assert_array_equal(predictor.predict.call_args.args[0], native)
    np.testing.assert_array_equal(
        registration.datastore.save_local_feature_predictor_image.call_args.args[0],
        probabilities,
    )
    saved = registration.datastore.save_local_feature_predictor_spots.call_args.args[0]
    # Centered 7x5x5 ROI: 175 voxels inside, 4x3x3=36 at either corner.
    np.testing.assert_array_equal(saved["sum_prob_pixels"], [43.75, 9, 9])
    # Exact sums of 100*z + 10*y + x + 1 over those integer coordinate boxes.
    np.testing.assert_array_equal(saved["sum_decon_pixels"], [77875, 5832, 26208])
    np.testing.assert_array_equal(
        saved[["tile_z_px", "tile_y_px", "tile_x_px"]],
        [[4, 4, 4], [0, 0, 0], [8, 8, 8]],
    )
    np.testing.assert_array_equal(saved["tile_idx"], 0)
    np.testing.assert_array_equal(saved["bit_idx"], 1)


@pytest.mark.unit
@pytest.mark.parametrize("deformable", [False, True])
@pytest.mark.parametrize("field_present", [False, True])
def test_tile_completion_requires_requested_sofima_field_without_reading_pixels(
    registration, deformable, field_present
):
    registration._decon_fiducial = False
    registration._perform_deformable_registration = deformable
    registration.datastore.load_local_round_transform_zyx_um.return_value = np.eye(4)
    field = Mock(shape=(3, 2, 3, 4))
    registration.datastore.load_local_sofima_flow_field.return_value = (
        (field, {"sofima_status": "ok"}) if field_present else None
    )

    assert registration._is_tile_complete("tile0000", process_readouts=False) == (
        not deformable or field_present
    )
    if deformable:
        registration.datastore.load_local_sofima_flow_field.assert_called_once_with(
            tile="tile0000", round="round002", return_future=None
        )
    else:
        registration.datastore.load_local_sofima_flow_field.assert_not_called()
    field.read.assert_not_called()


@pytest.mark.unit
def test_single_tile_global_metadata_preserves_origin_without_baking_in_camera(
    registration,
):
    registration._tile_ids = ["tile0000"]
    camera = np.diag([1, -1, 1, 1])
    registration.datastore.load_local_stage_position_zyx_um.return_value = (
        [10, 20, 30],
        camera,
    )
    registration.datastore.datastore_state = {"Corrected": True}

    registration.global_register(create_max_proj_tiff=False)

    saved = registration.datastore.save_global_coord_xforms_um.call_args.kwargs
    np.testing.assert_array_equal(saved["origin_zyx_um"], [10, 20, 30])
    np.testing.assert_array_equal(saved["affine_zyx_um"], np.eye(4))
    assert registration.datastore.datastore_state == {
        "Corrected": True,
        "GlobalRegistered": True,
    }


@pytest.mark.integration
def test_global_registration_saves_correction_and_composes_camera_only_for_fusion(
    registration, monkeypatch
):
    from multiview_stitcher import msi_utils, param_utils
    from multiview_stitcher import registration as mv_registration
    from multiview_stitcher import spatial_image_utils as si_utils

    camera = np.diag([1, -1, 1, 1])
    msims = [
        msi_utils.get_msim_from_sim(
            si_utils.get_sim_from_array(
                np.zeros((3, 8, 8), dtype=np.uint16),
                dims=("z", "y", "x"),
                scale={"z": 0.5, "y": 0.2, "x": 0.3},
                translation={"z": 10, "y": 20, "x": 30},
                affine=camera,
                transform_key="stage_metadata",
            ),
            scale_factors=[],
        )
        for _ in range(2)
    ]
    correction = np.eye(4)
    correction[:3, 3] = [1, 2, 3]

    def fit(views, **kwargs):
        for view in views:
            msi_utils.set_affine_transform(
                view,
                param_utils.affine_to_xaffine(correction),
                transform_key=kwargs["new_transform_key"],
                base_transform_key=kwargs["transform_key"],
            )
        return [correction, correction]

    monkeypatch.setattr(mv_registration, "register", fit)
    registration.load_global_fiducial_views = Mock(return_value=msims)
    registration.fuse_global_fiducial_views = Mock()

    registration.global_register(create_max_proj_tiff=False)

    expected = np.array([[1, 0, 0, 1], [0, -1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    for saved in registration.datastore.save_global_coord_xforms_um.call_args_list:
        np.testing.assert_array_equal(saved.kwargs["affine_zyx_um"], correction)
        np.testing.assert_array_equal(saved.kwargs["origin_zyx_um"], [10, 20, 30])
    for view in registration.fuse_global_fiducial_views.call_args.kwargs["msims"]:
        np.testing.assert_array_equal(
            msi_utils.get_transform_from_msim(view, "global_registered").data.squeeze(),
            expected,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "requested, present", [(False, False), (True, False), (True, True)]
)
def test_tile_completion_requires_requested_readout_deconvolution(
    registration, requested, present
):
    registration._decon_fiducial = False
    registration._perform_deformable_registration = False
    registration._decon_readout = requested
    registration.datastore.load_local_round_transform_zyx_um.return_value = np.eye(4)
    registration._has_valid_feature_predictor_outputs = Mock(return_value=True)
    registration._has_valid_deconvolved_readout_image = Mock(return_value=present)

    assert registration._is_tile_complete("tile0000") == (not requested or present)


@pytest.mark.integration
@pytest.mark.parametrize("axial_spacing", [0.315, 1.0, 1.5])
def test_global_registration_recovers_known_axial_and_lateral_tile_displacements(
    registration, axial_spacing
):
    from multiview_stitcher import msi_utils
    from multiview_stitcher import spatial_image_utils as si_utils
    from scipy.ndimage import gaussian_filter

    # Match the simulation's sampling range and use the standard registration
    # configuration. Crop a shared scene to create known tile displacements.
    volume = gaussian_filter(
        np.random.default_rng(41).random((54, 192, 288)), sigma=(2, 3, 3)
    )
    msims = []
    for z_start, x_start, stage_x in [(0, 0, 30.0), (3, 96, 30.0 + 84 * 0.081)]:
        sim = si_utils.get_sim_from_array(
            volume[z_start : z_start + 48, :, x_start : x_start + 192],
            dims=("z", "y", "x"),
            scale={"z": axial_spacing, "y": 0.081, "x": 0.081},
            translation={"z": 10, "y": 20, "x": stage_x},
            transform_key="stage_metadata",
        )
        msims.append(msi_utils.get_msim_from_sim(sim, scale_factors=[]))
    registration.load_global_fiducial_views = Mock(return_value=msims)
    registration.fuse_global_fiducial_views = Mock()

    registration.global_register(create_max_proj_tiff=False)

    saved = registration.datastore.save_global_coord_xforms_um.call_args_list[1].kwargs
    affine = saved["affine_zyx_um"]
    np.testing.assert_allclose(affine[:3, :3], np.eye(3), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(affine[3], [0, 0, 0, 1])
    # Three axial planes and twelve unrecorded lateral pixels. Check errors in
    # native pixels, independent of the physical sampling, to subpixel accuracy.
    np.testing.assert_allclose(
        affine[:3, 3] / [axial_spacing, 0.081, 0.081],
        [3, 0, 12],
        rtol=0,
        atol=0.25,
    )
