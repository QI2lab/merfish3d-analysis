import warnings

import numpy as np
import pytest


def _low_snr_shifted_fiducials() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create correlated broad signal with decorrelated high-frequency noise."""
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as ndi_shift

    shape = (24, 128, 128)
    true_pull_shift_px = np.asarray((3.0, 8.0, -6.0), dtype=np.float32)
    rng = np.random.default_rng(0)
    impulses = np.zeros(shape, dtype=np.float32)
    for _ in range(30):
        z, y, x = [rng.integers(3, size - 3) for size in shape]
        impulses[z, y, x] = rng.uniform(20, 100)
    signal = gaussian_filter(impulses, (1.5, 3.0, 3.0))

    # Consume the lower-noise realization used while isolating this regression.
    for noise_scale in (0.02, 0.05):
        fixed = signal + rng.normal(0, noise_scale, shape).astype(np.float32)
        moving = ndi_shift(
            signal,
            true_pull_shift_px,
            order=1,
            mode="constant",
        ) * np.float32(0.15)
        moving += rng.normal(0, noise_scale, shape).astype(np.float32)
    return fixed, moving, true_pull_shift_px


def _cucim_tiny_overlap_alias_images() -> tuple[np.ndarray, np.ndarray]:
    """Create images for which CuCIM prefers a one-plane periodic alias."""
    rng = np.random.default_rng(20260901)
    fixed = rng.normal(size=(16, 32, 32)).astype(np.float32)
    moving = np.empty_like(fixed)
    moving[0] = fixed[-1]
    moving[1:] = fixed[:-1] + rng.normal(
        scale=0.05,
        size=fixed[:-1].shape,
    ).astype(np.float32)
    return fixed, moving


def _require_cuda() -> object:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")
    return cp


def test_phase_shift_uses_maximum_overlap_alias_on_every_axis() -> None:
    from merfish3danalysis.utils.multiview_registration import (
        _maximum_overlap_phase_shift_px,
    )

    corrected = _maximum_overlap_phase_shift_px(
        (62.1, -2047.2, 2046.75),
        (63, 2048, 2048),
    )
    np.testing.assert_allclose(corrected, (-0.9, 0.8, -1.25), atol=1e-10)


def test_phase_candidate_selection_rejects_large_noise_peak_on_every_axis() -> None:
    from skimage.registration import phase_cross_correlation

    from merfish3danalysis.utils.multiview_registration import (
        _select_phase_correlation_pull_shift_px,
    )

    fixed, moving, true_pull_shift_px = _low_snr_shifted_fiducials()
    phase_only_pull = -phase_cross_correlation(
        fixed,
        moving,
        upsample_factor=10,
        disambiguate=False,
        normalization="phase",
    )[0]
    assert np.linalg.norm(phase_only_pull - true_pull_shift_px) > 20

    recovered_zyx = _select_phase_correlation_pull_shift_px(
        fixed,
        moving,
        phase_cross_correlation=phase_cross_correlation,
        array_module=np,
        to_numpy=np.asarray,
    )
    recovered_yx = _select_phase_correlation_pull_shift_px(
        fixed.max(axis=0),
        moving.max(axis=0),
        phase_cross_correlation=phase_cross_correlation,
        array_module=np,
        to_numpy=np.asarray,
    )

    np.testing.assert_allclose(recovered_zyx, true_pull_shift_px, atol=1.0)
    np.testing.assert_allclose(recovered_yx, true_pull_shift_px[1:], atol=1.5)


def test_cucim_disambiguation_selects_known_one_plane_alias() -> None:
    cp = _require_cuda()
    pytest.importorskip("cucim")
    from cucim.skimage.registration import phase_cross_correlation

    fixed, moving = _cucim_tiny_overlap_alias_images()
    periodic_shift = phase_cross_correlation(
        cp.asarray(fixed),
        cp.asarray(moving),
        upsample_factor=10,
        disambiguate=False,
    )[0]
    tiny_overlap_shift = phase_cross_correlation(
        cp.asarray(fixed),
        cp.asarray(moving),
        upsample_factor=10,
        disambiguate=True,
    )[0]

    np.testing.assert_allclose(cp.asnumpy(periodic_shift), (-1.0, 0.0, 0.0))
    np.testing.assert_allclose(cp.asnumpy(tiny_overlap_shift), (15.0, 0.0, 0.0))


def test_register_pair_to_fixed_rejects_known_one_plane_alias() -> None:
    _require_cuda()
    pytest.importorskip("cucim")
    from merfish3danalysis.utils.multiview_registration import register_pair_to_fixed

    fixed, moving = _cucim_tiny_overlap_alias_images()
    spacing_zyx_um = (0.32, 0.098, 0.098)

    transform = register_pair_to_fixed(
        fixed,
        moving,
        spacing_zyx_um=spacing_zyx_um,
    )

    recovered_shift_zyx_px = np.asarray(transform[:3, 3]) / np.asarray(spacing_zyx_um)
    np.testing.assert_allclose(recovered_shift_zyx_px, (1.0, 0.0, 0.0), atol=1e-5)


def test_register_pair_to_fixed_recovers_z_shift_for_warp_contract() -> None:
    _require_cuda()
    pytest.importorskip("cucim")
    from scipy.ndimage import shift as ndi_shift

    from merfish3danalysis.utils.multiview_registration import (
        register_pair_to_fixed,
        warp_array_to_reference_gpu,
    )

    shape = (31, 64, 64)
    z, y, x = np.indices(shape)
    fixed = np.zeros(shape, dtype=np.float32)
    for z0, y0, x0 in [(8, 18, 20), (16, 42, 35), (23, 30, 50)]:
        fixed += np.exp(
            -(((z - z0) / 2.0) ** 2 + ((y - y0) / 4.0) ** 2 + ((x - x0) / 4.0) ** 2)
        )

    true_shift_zyx_px = (2.0, 0.0, 0.0)
    moving = ndi_shift(
        fixed,
        shift=true_shift_zyx_px,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    spacing_zyx_um = (0.32, 0.098, 0.098)
    transform = register_pair_to_fixed(
        fixed,
        moving,
        spacing_zyx_um=spacing_zyx_um,
    )
    recovered_shift_zyx_px = np.asarray(transform[:3, 3]) / np.asarray(spacing_zyx_um)
    np.testing.assert_allclose(
        recovered_shift_zyx_px,
        true_shift_zyx_px,
        atol=0.25,
    )

    warped = warp_array_to_reference_gpu(
        moving,
        transform_zyx_um=transform,
        spacing_zyx_um=spacing_zyx_um,
        reference_shape=fixed.shape,
        gpu_id=0,
    )
    assert np.sqrt(np.mean((warped - fixed) ** 2)) < 1e-3


def test_register_pair_to_fixed_residual_uses_valid_lateral_overlap() -> None:
    _require_cuda()
    pytest.importorskip("cucim")
    from scipy.ndimage import shift as ndi_shift

    from merfish3danalysis.utils.multiview_registration import register_pair_to_fixed

    shape = (24, 48, 48)
    z, y, x = np.indices(shape)
    fixed = np.zeros(shape, dtype=np.float32)
    for z0, y0, x0 in [(7, 12, 14), (12, 30, 27), (18, 22, 38)]:
        fixed += np.exp(
            -(((z - z0) / 1.8) ** 2 + ((y - y0) / 3.2) ** 2 + ((x - x0) / 3.2) ** 2)
        )

    true_shift_zyx_px = (1.0, 8.0, -7.0)
    moving = ndi_shift(
        fixed,
        shift=true_shift_zyx_px,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    spacing_zyx_um = (0.32, 0.098, 0.098)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        transform = register_pair_to_fixed(
            fixed,
            moving,
            spacing_zyx_um=spacing_zyx_um,
        )

    recovered_shift_zyx_px = np.asarray(transform[:3, 3]) / np.asarray(spacing_zyx_um)
    np.testing.assert_allclose(
        recovered_shift_zyx_px,
        true_shift_zyx_px,
        atol=0.35,
    )
