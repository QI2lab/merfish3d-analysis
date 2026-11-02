from queue import SimpleQueue
from unittest.mock import Mock

import numpy as np
import pytest

from merfish3danalysis.DataRegistration import (
    DataRegistration,
    GlobalRegistrationConfig,
)


@pytest.mark.unit
def test_registration_resumes_completed_scene_images(image_datastore):
    from yaozarrs import open_group

    datastore = image_datastore
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    for tile in range(datastore.num_tiles):
        for round_index in range(datastore.num_rounds):
            datastore.save_local_corrected_image(pixels, tile=tile, round=round_index)
            datastore.save_local_deconvolved_fiducial_image(
                pixels, tile=tile, round=round_index
            )
            datastore.save_local_round_transform_zyx_um(
                np.eye(4), tile=tile, round=round_index
            )
    before = open_group(datastore.datastore_path).attrs["qi2lab"]["products"]
    registration = DataRegistration(
        datastore,
        num_gpus=0,
        decon_fiducial=True,
        perform_deformable_registration=False,
        overwrite_outputs=False,
        verbose=0,
    )
    registration.register_all_tiles(process_readouts=False)
    assert open_group(datastore.datastore_path).attrs["qi2lab"]["products"] == before


@pytest.mark.unit
@pytest.mark.parametrize("psf_idx", [0, 1, 2])
def test_fiducial_deconvolution_uses_recorded_psf(registration, monkeypatch, psf_idx):
    import merfish3danalysis.DataRegistration as registration_module

    registration.datastore.channel_psfs = np.eye(3, dtype=np.float32).reshape(
        3, 1, 1, 3
    )
    registration = DataRegistration(
        registration.datastore, decon_fiducial=True, verbose=0
    )
    registration.tile_id = 0
    native = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    registration.datastore.save_local_corrected_image(
        native, tile=0, round=0, psf_idx=psf_idx
    )
    deconvolve = Mock(return_value=native.astype(np.float32))
    monkeypatch.setattr(
        registration_module, "_run_chunked_rlgc_remembering_crop", deconvolve
    )

    result = registration_module._load_deconvolve_fiducial_round(
        registration, "round001", 0, Mock()
    )

    np.testing.assert_array_equal(
        deconvolve.call_args.kwargs["psf"], registration._psfs[psf_idx]
    )
    np.testing.assert_array_equal(result, native)
    np.testing.assert_array_equal(
        registration.datastore.load_local_deconvolved_fiducial_image(
            tile=0, round=0, return_future=False
        ),
        native,
    )


@pytest.mark.unit
@pytest.mark.gpu
def test_fiducial_deconvolution_restores_generated_point_source(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires CUDA")

    from scipy.ndimage import convolve

    from merfish3danalysis.DataRegistration import _load_deconvolve_fiducial_round
    from merfish3danalysis.qi2labDataStore import qi2labDataStore
    from merfish3danalysis.utils.rlgc import chunked_rlgc

    # A normalized anisotropic PSF conserves photons when imaging a point
    # source. The unused identity PSF must not replace its recorded calibration.
    z, y, x = np.indices((5, 5, 5)) - 2
    psf = np.exp(-0.5 * ((z / 0.8) ** 2 + y**2 + (x / 1.2) ** 2))
    psf = (psf / psf.sum()).astype(np.float32)
    identity_psf = np.zeros_like(psf)
    identity_psf[2, 2, 2] = 1
    truth = np.zeros((9, 21, 25), dtype=np.float32)
    center = (4, 10, 12)
    truth[center] = 10000
    native = np.rint(convolve(truth, psf, mode="constant")).astype(np.uint16)

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.num_tiles = 1
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.experiment_order = np.array([[1, 1]])
    datastore.voxel_size_zyx_um = [0.315, 0.081, 0.081]
    datastore.channel_psfs = np.stack([identity_psf, psf])
    datastore.initialize_tile(0)
    datastore.save_local_corrected_image(native, tile=0, round=0, psf_idx=1)
    registration = DataRegistration(datastore, decon_fiducial=True, verbose=0)
    registration.tile_id = 0

    recovered = _load_deconvolve_fiducial_round(
        registration, "round001", 0, chunked_rlgc
    )

    assert np.unravel_index(np.argmax(recovered), recovered.shape) == center
    assert recovered[center] > 1.5 * native[center]
    assert recovered.sum() == pytest.approx(truth.sum(), rel=0.05)
    assert np.abs(recovered.astype(float) - truth).sum() < np.abs(native - truth).sum()
    # The recovered photons must still explain the acquired image through the
    # independently specified forward optical model (5% relative L1 residual).
    forward = convolve(recovered.astype(float), psf, mode="constant")
    assert np.abs(forward - native).sum() / native.sum() < 0.05
    np.testing.assert_array_equal(
        datastore.load_local_deconvolved_fiducial_image(
            tile=0, round=0, return_future=False
        ),
        recovered,
    )
    np.testing.assert_array_equal(
        datastore.load_local_corrected_image(tile=0, round=0, return_future=False),
        native,
    )


@pytest.fixture
def registration(image_datastore):
    return DataRegistration(
        datastore=image_datastore,
        verbose=0,
        global_registration_config=GlobalRegistrationConfig(fusion_workers=1),
    )


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
    import pandas as pd

    registration._global_registration = global_registration
    registration._perform_deformable_registration = False
    registration._num_gpus = 0
    image = np.ones((2, 3, 4), dtype=np.uint16)
    for tile in range(registration.datastore.num_tiles):
        for round_index in range(registration.datastore.num_rounds):
            registration.datastore.save_local_corrected_image(
                image, tile=tile, round=round_index
            )
            registration.datastore.save_local_deconvolved_fiducial_image(
                image, tile=tile, round=round_index
            )
            registration.datastore.save_local_round_transform_zyx_um(
                np.eye(4), tile=tile, round=round_index
            )
        if process_readouts:
            for bit in range(registration.datastore.num_bits):
                registration.datastore.save_local_corrected_image(
                    image, tile=tile, bit=bit
                )
                registration.datastore.save_local_deconvolved_readout_image(
                    image, tile=tile, bit=bit
                )
                registration.datastore.save_local_feature_predictor_image(
                    image, tile=tile, bit=bit
                )
                registration.datastore.save_local_feature_predictor_spots(
                    pd.DataFrame({"z": [0], "y": [0], "x": [0]}), tile=tile, bit=bit
                )
    registration.global_register = Mock()
    registration.register_all_tiles(process_readouts=process_readouts)
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


@pytest.mark.unit
@pytest.mark.gpu
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
    registration._decon_fiducial = False
    registration._perform_deformable_registration = False
    registration.datastore.save_local_corrected_image(
        np.rint(moving * 10000).astype(np.uint16), tile=0, round=1
    )
    fixed = np.rint(fixed * 10000).astype(np.uint16)
    moving = np.rint(moving * 10000).astype(np.uint16)
    # Keep the test process's device visibility unchanged; execute real CUDA math.
    monkeypatch.setattr(
        registration_module, "_restrict_worker_to_assigned_gpu", Mock(return_value=0)
    )
    results = SimpleQueue()

    registration_module._process_fiducial_rounds_on_gpu(
        registration, ["round002"], fixed, 0, results
    )

    transform = registration.datastore.load_local_round_transform_zyx_um(
        tile=0, round=1
    )
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), rtol=0, atol=1e-7)
    # Pull mapping into the displaced native image, in micrometers.
    np.testing.assert_allclose(transform[:3, 3], [0.8, 0.6, -1.6], rtol=0, atol=0.05)
    np.testing.assert_array_equal(transform[3], [0, 0, 0, 1])
    restored = warp_array_to_reference_gpu(
        moving,
        transform_zyx_um=transform,
        spacing_zyx_um=registration.datastore.voxel_size_zyx_um,
        reference_shape=fixed.shape,
    )
    # Ignore boundary samples lost in the explicitly translated native image.
    np.testing.assert_allclose(
        restored[4:-4, 6:-6, 8:-8], fixed[4:-4, 6:-6, 8:-8], atol=1
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


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("deconvolve", [False, True])
@pytest.mark.parametrize("excitation_um", [0.55, 0.70])
def test_readout_worker_preserves_native_pixels_and_clips_spot_rois_at_edges(
    registration, monkeypatch, deconvolve, excitation_um
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
    registration.datastore.channel_psfs = np.eye(3, dtype=np.float32).reshape(
        3, 1, 1, 3
    )
    registration = DataRegistration(
        registration.datastore, decon_readout=deconvolve, verbose=0
    )
    registration.tile_id = 0
    registration.datastore.save_local_round_linker(1, tile=0, bit=0)
    registration.datastore.save_local_wavelengths_um(
        (excitation_um, excitation_um + 0.03), tile=0, bit=0
    )
    deconvolver = Mock(return_value=native.astype(np.float32))
    monkeypatch.setattr(
        registration_module, "_run_chunked_rlgc_remembering_crop", deconvolver
    )
    registration.datastore.save_local_corrected_image(native, tile=0, bit=0, psf_idx=1)

    registration_module._apply_bits_on_gpu(registration, ["bit001"], 0)

    if deconvolve:
        np.testing.assert_array_equal(
            deconvolver.call_args.kwargs["psf"], registration._psfs[1]
        )
        np.testing.assert_array_equal(
            registration.datastore.load_local_deconvolved_readout_image(
                tile=0, bit=0, return_future=False
            ),
            native,
        )
    else:
        deconvolver.assert_not_called()
        assert (
            registration.datastore.load_local_deconvolved_readout_image(
                tile=0, bit=0, return_future=False
            )
            is None
        )
    np.testing.assert_array_equal(predictor.predict.call_args.args[0], native)
    np.testing.assert_array_equal(
        registration.datastore.load_local_feature_predictor_image(
            tile=0, bit=0, return_future=False
        ),
        probabilities,
    )
    saved = registration.datastore.load_local_feature_predictor_spots(tile=0, bit=0)
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
    for round_index in range(2):
        registration.datastore.save_local_round_transform_zyx_um(
            np.eye(4), tile=0, round=round_index
        )
    if field_present:
        shape = (2, 3, 4)
        registration.datastore.save_local_sofima_flow_field(
            np.zeros((3, *shape), dtype=np.float32),
            tile=0,
            round=1,
            reference_round=0,
            map_stride_zyx_px=(1, 1, 1),
            map_box_start_xyz_px=(0, 0, 0),
            map_box_size_xyz_px=shape[::-1],
            reference_shape_zyx_px=shape,
            moving_shape_zyx_px=shape,
        )
    assert registration._is_tile_complete("tile0000", process_readouts=False) == (
        not deformable or field_present
    )


@pytest.mark.unit
def test_single_tile_global_metadata_preserves_origin_without_baking_in_camera(
    registration,
):
    registration.datastore.num_tiles = 1
    registration = DataRegistration(registration.datastore, verbose=0)
    camera = np.diag([1, -1, 1, 1])
    registration.datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], camera, tile=0, round=0
    )
    registration.global_register(create_max_proj_tiff=False)
    affine, origin, _ = registration.datastore.load_global_coord_xforms_um(tile=0)
    np.testing.assert_array_equal(origin, [10, 20, 30])
    np.testing.assert_array_equal(affine, np.eye(4))
    assert registration.datastore.datastore_state["GlobalRegistered"]


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
    import pandas as pd

    image = np.ones((2, 3, 4), dtype=np.uint16)
    for round_index in range(2):
        registration.datastore.save_local_round_transform_zyx_um(
            np.eye(4), tile=0, round=round_index
        )
    for bit in range(4):
        registration.datastore.save_local_corrected_image(image, tile=0, bit=bit)
        registration.datastore.save_local_feature_predictor_image(
            image, tile=0, bit=bit
        )
        registration.datastore.save_local_feature_predictor_spots(
            pd.DataFrame({"z": [0], "y": [0], "x": [0]}), tile=0, bit=bit
        )
        if present:
            registration.datastore.save_local_deconvolved_readout_image(
                image, tile=0, bit=bit
            )
    assert registration._is_tile_complete("tile0000") == (not requested or present)


@pytest.mark.integration
@pytest.mark.parametrize("axial_spacing", [0.315, 1.0, 1.5])
def test_global_registration_recovers_known_axial_and_lateral_tile_displacements(
    registration, axial_spacing
):
    from scipy.ndimage import gaussian_filter

    registration.datastore.voxel_size_zyx_um = [axial_spacing, 0.081, 0.081]
    volume = gaussian_filter(
        np.random.default_rng(41).random((54, 192, 288)), sigma=(2, 3, 3)
    )
    for tile, (z_start, x_start, stage_x) in enumerate(
        [(0, 0, 30.0), (3, 96, 30.0 + 84 * 0.081)]
    ):
        image = (
            volume[z_start : z_start + 48, :, x_start : x_start + 192] * 60000
        ).astype(np.uint16)
        registration.datastore.save_local_corrected_image(image, tile=tile, round=0)
        registration.datastore.save_local_stage_position_zyx_um(
            [10, 20, stage_x], np.eye(4), tile=tile, round=0
        )
    registration._decon_fiducial = False
    registration.global_register(create_max_proj_tiff=False)
    affine, _, _ = registration.datastore.load_global_coord_xforms_um(tile=1)
    np.testing.assert_allclose(affine[:3, :3], np.eye(3), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(affine[3], [0, 0, 0, 1])
    np.testing.assert_allclose(
        affine[:3, 3] / [axial_spacing, 0.081, 0.081],
        [3, 0, 12],
        rtol=0,
        atol=0.25,
    )
