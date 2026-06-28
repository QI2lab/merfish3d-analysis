import warnings
from pathlib import Path

import numpy as np
import pytest


def _flow_to_rgb(flow_xyz: np.ndarray, max_abs_flow: float) -> np.ndarray:
    """
    Convert XYZ flow components into an RGB image.

    Parameters
    ----------
    flow_xyz : numpy.ndarray
        Flow field in ``(3, z, y, x)`` order.
    max_abs_flow : float
        Symmetric display range in pixels.

    Returns
    -------
    numpy.ndarray
        RGB flow visualization in ``(z, y, x, 3)`` order.
    """

    normalized = 0.5 + 0.5 * flow_xyz / max(float(max_abs_flow), 1e-6)
    return np.moveaxis(np.clip(normalized, 0.0, 1.0), 0, -1)


def _save_sofima_recovery_png(
    output_path: Path,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    corrected: np.ndarray,
    expected_flow_xyz: np.ndarray,
    estimated_flow_xyz: np.ndarray,
) -> None:
    """
    Save a multi-plane SOFIMA recovery diagnostic image.

    Parameters
    ----------
    output_path : pathlib.Path
        PNG path to write.
    fixed : numpy.ndarray
        Reference image in Z, Y, X order.
    moving : numpy.ndarray
        Moving image before SOFIMA correction.
    corrected : numpy.ndarray
        Moving image after SOFIMA correction.
    expected_flow_xyz : numpy.ndarray
        Ground-truth flow on the estimator grid in ``(3, z, y, x)`` order.
    estimated_flow_xyz : numpy.ndarray
        Recovered flow on the estimator grid in ``(3, z, y, x)`` order.

    Returns
    -------
    None
        The PNG is written to ``output_path``.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_abs_flow = float(
        max(np.max(np.abs(expected_flow_xyz)), np.max(np.abs(estimated_flow_xyz)))
    )
    expected_rgb = _flow_to_rgb(expected_flow_xyz, max_abs_flow)
    estimated_rgb = _flow_to_rgb(estimated_flow_xyz, max_abs_flow)
    squared_error = (fixed - corrected) ** 2
    image_vmin = float(min(np.min(fixed), np.min(moving), np.min(corrected)))
    image_vmax = float(max(np.max(fixed), np.max(moving), np.max(corrected)))
    error_vmax = float(np.percentile(squared_error, 99.5))
    z_planes = np.linspace(0, fixed.shape[0] - 1, num=5, dtype=int)

    columns = (
        "reference",
        "expected flow RGB=X/Y/Z",
        "moving",
        "recovered flow RGB=X/Y/Z",
        "corrected",
        "squared error",
    )
    fig, axes = plt.subplots(
        len(z_planes),
        len(columns),
        figsize=(3.0 * len(columns), 2.6 * len(z_planes)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, z_plane in enumerate(z_planes):
        flow_z = round(
            z_plane * (estimated_flow_xyz.shape[1] - 1) / (fixed.shape[0] - 1)
        )
        panels = (
            (fixed[z_plane], "gray", image_vmin, image_vmax),
            (expected_rgb[flow_z], None, None, None),
            (moving[z_plane], "gray", image_vmin, image_vmax),
            (estimated_rgb[flow_z], None, None, None),
            (corrected[z_plane], "gray", image_vmin, image_vmax),
            (squared_error[z_plane], "magma", 0.0, error_vmax),
        )
        for column_index, (panel, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_index, column_index]
            ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(columns[column_index], fontsize=10)
            if column_index == 0:
                ax.set_ylabel(f"z={int(z_plane)}", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def test_register_pair_to_fixed_recovers_z_shift_for_warp_contract() -> None:
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    from scipy.ndimage import shift as ndi_shift

    from merfish3danalysis.utils.multiview_registration import (
        register_pair_to_fixed,
        warp_array_to_reference_gpu,
    )

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")

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
        registration_binning={"z": 1, "y": 3, "x": 3},
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
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cucim")
    from scipy.ndimage import shift as ndi_shift

    from merfish3danalysis.utils.multiview_registration import register_pair_to_fixed

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")

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


def test_sofima_estimator_recovers_object_model_warp_field(
    request: pytest.FixtureRequest,
) -> None:
    cp = pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from scipy.ndimage import gaussian_filter, map_coordinates

    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_with_affine_and_sofima_flow_gpu,
    )
    from merfish3danalysis.utils.sofima_registration import (
        estimate_sofima_flow_field_xyz_px,
    )

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")

    upsample = 3
    camera_shape_zyx = (32, 128, 128)
    highres_shape_zyx = tuple(axis_size * upsample for axis_size in camera_shape_zyx)
    patch_size_zyx = (10, 32, 32)
    stride_zyx = (5.0, 16.0, 16.0)
    box_start_xyz = (
        float(patch_size_zyx[2]) / 2.0,
        float(patch_size_zyx[1]) / 2.0,
        float(patch_size_zyx[0]) / 2.0,
    )

    def _make_objects() -> tuple[
        np.ndarray,
        np.ndarray,
        list[tuple[np.ndarray, np.float32, np.ndarray]],
    ]:
        rng = np.random.default_rng(0)
        margin = np.asarray((8, 18, 18), dtype=np.float32)
        highres_shape = np.asarray(highres_shape_zyx, dtype=np.float32)
        point_count = 3500
        points_zyx = rng.uniform(
            margin,
            highres_shape - margin,
            size=(point_count, 3),
        ).astype(np.float32)
        point_weights = rng.uniform(10, 100, point_count).astype(np.float32)
        blobs = []
        for _ in range(80):
            center_zyx = rng.uniform(margin, highres_shape - margin).astype(np.float32)
            amplitude = np.float32(rng.uniform(30, 100))
            sigma_zyx = np.asarray(
                [
                    rng.uniform(3.0, 7.0),
                    rng.uniform(4.0, 10.0),
                    rng.uniform(4.0, 10.0),
                ],
                dtype=np.float32,
            )
            blobs.append((center_zyx, amplitude, sigma_zyx))
        return points_zyx, point_weights, blobs

    def _splat_points(
        volume: np.ndarray,
        points_zyx: np.ndarray,
        point_weights: np.ndarray,
    ) -> None:
        shape = np.asarray(volume.shape)
        base = np.floor(points_zyx).astype(np.int32)
        fraction = points_zyx - base
        for dz in (0, 1):
            weight_z = (1 - fraction[:, 0]) if dz == 0 else fraction[:, 0]
            zz = base[:, 0] + dz
            for dy in (0, 1):
                weight_y = (1 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
                yy = base[:, 1] + dy
                for dx in (0, 1):
                    weight_x = (1 - fraction[:, 2]) if dx == 0 else fraction[:, 2]
                    xx = base[:, 2] + dx
                    mask = (
                        (zz >= 0)
                        & (yy >= 0)
                        & (xx >= 0)
                        & (zz < shape[0])
                        & (yy < shape[1])
                        & (xx < shape[2])
                    )
                    np.add.at(
                        volume,
                        (zz[mask], yy[mask], xx[mask]),
                        point_weights[mask]
                        * weight_z[mask]
                        * weight_y[mask]
                        * weight_x[mask],
                    )

    def _render_highres_objects(
        points_zyx: np.ndarray,
        point_weights: np.ndarray,
        blobs: list[tuple[np.ndarray, np.float32, np.ndarray]],
    ) -> np.ndarray:
        rng = np.random.default_rng(1)
        objects = np.zeros(highres_shape_zyx, dtype=np.float32)
        _splat_points(objects, points_zyx, point_weights)
        for center_zyx, amplitude, sigma_zyx in blobs:
            samples = center_zyx + rng.normal(0.0, sigma_zyx, size=(96, 3)).astype(
                np.float32
            )
            _splat_points(
                objects,
                samples,
                np.full(96, amplitude / 96.0, dtype=np.float32),
            )
        return objects

    def _image_camera_from_objects(
        points_zyx: np.ndarray,
        point_weights: np.ndarray,
        blobs: list[tuple[np.ndarray, np.float32, np.ndarray]],
    ) -> np.ndarray:
        highres_image = _render_highres_objects(points_zyx, point_weights, blobs) + 1.0
        highres_image = gaussian_filter(highres_image, sigma=(3.6, 3.3, 3.3))
        z_subsampled = highres_image[upsample // 2 :: upsample]
        z_size, y_size, x_size = z_subsampled.shape
        camera = z_subsampled.reshape(
            z_size,
            y_size // upsample,
            upsample,
            x_size // upsample,
            upsample,
        ).mean(axis=(2, 4))
        return camera.astype(np.float32)

    def _make_camera_flow(amplitude: float) -> np.ndarray:
        flow_shape_zyx = tuple(
            int(np.floor((axis_size - patch_size) / stride)) + 1
            for axis_size, patch_size, stride in zip(
                camera_shape_zyx,
                patch_size_zyx,
                stride_zyx,
                strict=False,
            )
        )
        z_grid, y_grid, x_grid = np.indices(flow_shape_zyx, dtype=np.float32)
        z_center = (flow_shape_zyx[0] - 1) / 2
        y_center = (flow_shape_zyx[1] - 1) / 2
        x_center = (flow_shape_zyx[2] - 1) / 2
        x_norm = (x_grid - x_center) / max(x_center, 1)
        y_norm = (y_grid - y_center) / max(y_center, 1)
        z_norm = (z_grid - z_center) / max(z_center, 1)
        flow_xyz = np.zeros((3, *flow_shape_zyx), dtype=np.float32)
        flow_xyz[0] = amplitude * (0.8 * x_norm + 0.2 * y_norm)
        flow_xyz[1] = amplitude * (-0.7 * y_norm + 0.15 * x_norm)
        flow_xyz[2] = amplitude * (0.4 * z_norm + 0.05 * x_norm)
        return flow_xyz

    def _interpolate_flow_at_camera_points(
        flow_xyz: np.ndarray,
        points_camera_zyx: np.ndarray,
    ) -> np.ndarray:
        coords = np.stack(
            [
                (points_camera_zyx[:, 0] - box_start_xyz[2]) / stride_zyx[0],
                (points_camera_zyx[:, 1] - box_start_xyz[1]) / stride_zyx[1],
                (points_camera_zyx[:, 2] - box_start_xyz[0]) / stride_zyx[2],
            ],
            axis=0,
        )
        displacement_xyz = np.stack(
            [
                map_coordinates(
                    flow_xyz[channel_index], coords, order=1, mode="nearest"
                )
                for channel_index in range(3)
            ],
            axis=1,
        )
        return displacement_xyz[:, [2, 1, 0]].astype(np.float32)

    def _deform_objects(
        points_zyx: np.ndarray,
        blobs: list[tuple[np.ndarray, np.float32, np.ndarray]],
        flow_xyz: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.float32, np.ndarray]]]:
        displacement_zyx = _interpolate_flow_at_camera_points(
            flow_xyz,
            points_zyx / upsample,
        )
        moved_points_zyx = points_zyx + displacement_zyx * upsample
        moved_blobs = []
        for center_zyx, amplitude, sigma_zyx in blobs:
            center_camera_zyx = (center_zyx / upsample)[np.newaxis, :]
            center_displacement_zyx = _interpolate_flow_at_camera_points(
                flow_xyz,
                center_camera_zyx,
            )[0]
            moved_blobs.append(
                (center_zyx + center_displacement_zyx * upsample, amplitude, sigma_zyx)
            )
        return moved_points_zyx.astype(np.float32), moved_blobs

    def _expected_flow_on_estimated_grid(
        estimated_flow_xyz: np.ndarray,
        metadata: dict[str, object],
        known_flow_xyz: np.ndarray,
    ) -> np.ndarray:
        estimated_stride_zyx = tuple(float(v) for v in metadata["map_stride_zyx_px"])
        estimated_box_start_xyz = tuple(
            float(v) for v in metadata["map_box_start_xyz_px"]
        )
        z_grid, y_grid, x_grid = np.indices(
            estimated_flow_xyz.shape[1:],
            dtype=np.float32,
        )
        coords = np.stack(
            [
                (
                    estimated_box_start_xyz[2]
                    + z_grid * estimated_stride_zyx[0]
                    - box_start_xyz[2]
                )
                / stride_zyx[0],
                (
                    estimated_box_start_xyz[1]
                    + y_grid * estimated_stride_zyx[1]
                    - box_start_xyz[1]
                )
                / stride_zyx[1],
                (
                    estimated_box_start_xyz[0]
                    + x_grid * estimated_stride_zyx[2]
                    - box_start_xyz[0]
                )
                / stride_zyx[2],
            ],
            axis=0,
        )
        return np.stack(
            [
                map_coordinates(
                    known_flow_xyz[channel_index],
                    coords,
                    order=1,
                    mode="nearest",
                )
                for channel_index in range(3)
            ],
            axis=0,
        )

    points_zyx, point_weights, blobs = _make_objects()
    fixed = _image_camera_from_objects(points_zyx, point_weights, blobs)
    known_flow_xyz = _make_camera_flow(10.0)
    moved_points_zyx, moved_blobs = _deform_objects(points_zyx, blobs, known_flow_xyz)
    moving = _image_camera_from_objects(moved_points_zyx, point_weights, moved_blobs)

    estimated_flow_xyz, metadata = estimate_sofima_flow_field_xyz_px(
        fixed,
        moving,
    )

    expected_flow_xyz = _expected_flow_on_estimated_grid(
        estimated_flow_xyz,
        metadata,
        known_flow_xyz,
    )
    flow_error_xyz = estimated_flow_xyz - expected_flow_xyz
    flow_rmse_xyz = np.sqrt(np.mean(flow_error_xyz**2, axis=(1, 2, 3)))
    interior_flow_error_xyz = flow_error_xyz[
        :,
        1:-1,
        1:-1,
        1:-1,
    ]
    output_png = request.config.getoption("--sofima-output-png")
    if output_png is not None:
        corrected = warp_array_to_reference_with_affine_and_sofima_flow_gpu(
            moving,
            transform_zyx_um=np.eye(4, dtype=np.float32),
            spacing_zyx_um=(1.0, 1.0, 1.0),
            reference_shape=fixed.shape,
            sofima_flow_field_xyz_px=estimated_flow_xyz,
            flow_field_stride_zyx_px=metadata["map_stride_zyx_px"],
            flow_field_box_start_xyz_px=metadata["map_box_start_xyz_px"],
            mode="nearest",
            gpu_id=0,
        ).astype(np.float32)
        _save_sofima_recovery_png(
            Path(output_png),
            fixed=fixed,
            moving=moving,
            corrected=corrected,
            expected_flow_xyz=expected_flow_xyz,
            estimated_flow_xyz=estimated_flow_xyz,
        )

    assert metadata["status"] == "ok"
    assert metadata["valid_flow_vectors"] > 400
    assert metadata["residual_iterations"] == 2
    assert estimated_flow_xyz.shape[0] == 3
    assert np.sqrt(np.mean(flow_error_xyz**2)) < 0.75
    assert np.sqrt(np.mean(interior_flow_error_xyz**2)) < 0.75
    assert np.all(flow_rmse_xyz < np.asarray([0.7, 0.7, 0.75]))
    assert np.min(estimated_flow_xyz[0]) <= -9.5
    assert np.max(estimated_flow_xyz[0]) >= 9.5
    assert np.min(estimated_flow_xyz[1]) <= -7.5
    assert np.max(estimated_flow_xyz[1]) >= 7.5
    assert np.min(estimated_flow_xyz[2]) <= -4.5
    assert np.max(estimated_flow_xyz[2]) >= 3.5


def test_sofima_flow_field_datastore_roundtrip_preserves_warp(tmp_path) -> None:
    cp = pytest.importorskip("cupy")
    from merfish3danalysis.qi2labDataStore import qi2labDataStore
    from merfish3danalysis.utils.multiview_registration import (
        warp_array_to_reference_with_affine_and_sofima_flow_gpu,
    )

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is not available.")

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore", validate=False)
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.voxel_size_zyx_um = (0.3, 0.1, 0.1)
    datastore.num_tiles = 1
    datastore.num_rounds = 2
    datastore.experiment_order = np.asarray([[1, 1], [2, 1]], dtype=np.int64)

    rng = np.random.default_rng(13)
    image = rng.normal(size=(6, 11, 13)).astype(np.float32)
    flow_field_xyz_px = np.zeros((3, 2, 3, 4), dtype=np.float32)
    flow_field_xyz_px[0] = np.linspace(
        -0.25,
        0.5,
        flow_field_xyz_px[0].size,
        dtype=np.float32,
    ).reshape(flow_field_xyz_px.shape[1:])
    flow_field_xyz_px[1] = np.linspace(
        0.4,
        -0.2,
        flow_field_xyz_px[1].size,
        dtype=np.float32,
    ).reshape(flow_field_xyz_px.shape[1:])
    flow_field_xyz_px[2] = np.float32(0.125)
    map_stride_zyx_px = (2.0, 4.0, 4.0)
    map_box_start_xyz_px = (2.0, 2.0, 1.0)
    map_box_size_xyz_px = (13.0, 11.0, 6.0)

    in_memory_warp = warp_array_to_reference_with_affine_and_sofima_flow_gpu(
        image,
        transform_zyx_um=np.eye(4, dtype=np.float32),
        spacing_zyx_um=datastore.voxel_size_zyx_um,
        reference_shape=image.shape,
        sofima_flow_field_xyz_px=flow_field_xyz_px,
        flow_field_stride_zyx_px=map_stride_zyx_px,
        flow_field_box_start_xyz_px=map_box_start_xyz_px,
        mode="nearest",
        gpu_id=0,
    )
    datastore.save_local_sofima_flow_field(
        flow_field_xyz_px,
        tile=0,
        round=1,
        reference_round=0,
        map_stride_zyx_px=map_stride_zyx_px,
        map_box_start_xyz_px=map_box_start_xyz_px,
        map_box_size_xyz_px=map_box_size_xyz_px,
        reference_shape_zyx_px=image.shape,
        moving_shape_zyx_px=image.shape,
        sofima_status="ok",
        valid_flow_vectors=int(np.prod(flow_field_xyz_px.shape[1:])),
        return_future=False,
    )

    loaded = datastore.load_local_sofima_flow_field(
        tile=0,
        round=1,
        return_future=False,
    )
    assert loaded is not None
    loaded_flow_field_xyz_px, loaded_attrs = loaded
    np.testing.assert_array_equal(loaded_flow_field_xyz_px, flow_field_xyz_px)
    assert loaded_attrs["flow_channel_order"] == "xyz"
    assert loaded_attrs["flow_spatial_order"] == "zyx"
    np.testing.assert_allclose(
        loaded_attrs["map_stride_zyx_px"],
        map_stride_zyx_px,
    )
    np.testing.assert_allclose(
        loaded_attrs["map_box_start_xyz_px"],
        map_box_start_xyz_px,
    )
    np.testing.assert_allclose(
        loaded_attrs["map_box_size_xyz_px"],
        map_box_size_xyz_px,
    )

    reloaded_warp = warp_array_to_reference_with_affine_and_sofima_flow_gpu(
        image,
        transform_zyx_um=np.eye(4, dtype=np.float32),
        spacing_zyx_um=datastore.voxel_size_zyx_um,
        reference_shape=image.shape,
        sofima_flow_field_xyz_px=loaded_flow_field_xyz_px,
        flow_field_stride_zyx_px=loaded_attrs["map_stride_zyx_px"],
        flow_field_box_start_xyz_px=loaded_attrs["map_box_start_xyz_px"],
        mode="nearest",
        gpu_id=0,
    )
    np.testing.assert_array_equal(reloaded_warp, in_memory_warp)
