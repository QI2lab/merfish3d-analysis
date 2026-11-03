"""Keep the fused segmentation grid, TIFF calibration, and ROI mapping aligned."""

from xml.etree import ElementTree

import numpy as np
import pytest
from roifile import roiread, roiwrite
from tifffile import TiffFile
from yaozarrs import open_group

import merfish3danalysis.DataRegistration as registration_module
from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.integration
@pytest.mark.parametrize("image_datastore", ["one_tile"], indirect=True)
@pytest.mark.parametrize(
    "native_spacing, expected_spacing",
    [
        ([0.315, 0.081, 0.081], [0.315, 0.316, 0.316]),
        ([0.32, 0.098, 0.112], [0.32, 0.323, 0.325]),
    ],
    ids=["simulation-nyquist", "anisotropic-xy"],
)
def test_fiducial_fusion_tiff_and_rois_share_downsampled_grid(
    tmp_path, image_datastore, native_spacing, expected_spacing
):
    datastore = image_datastore
    datastore.voxel_size_zyx_um = native_spacing
    data = np.random.default_rng(8).integers(0, 65536, (7, 140, 168), dtype=np.uint16)
    datastore.save_local_stage_position_zyx_um(
        [12, 100, 200], np.eye(4), tile=0, round=0
    )
    datastore.save_local_corrected_image(data, tile=0, round=0)
    registration = registration_module.DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=registration_module.GlobalRegistrationConfig(
            fusion_workers=1
        ),
    )
    registration.fuse_global_registered(create_max_proj_tiff=True)

    fused_path = datastore.fused_image_path()
    attributes = datastore.load_image_metadata(fused_path)
    assert attributes["spacing_zyx_um"] == expected_spacing
    np.testing.assert_allclose(attributes["origin_zyx_um"], [12, 100, 200])
    np.testing.assert_array_equal(attributes["affine_zyx_um"], np.eye(4))
    group = open_group(fused_path)
    dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0].path
    fused = group[dataset].to_tensorstore()
    assert fused.shape[-3] == data.shape[0]
    for actual, native_size, native_step, fused_step in zip(
        fused.shape[-2:],
        data.shape[-2:],
        native_spacing[1:],
        expected_spacing[1:],
        strict=True,
    ):
        assert abs(actual - native_size * native_step / fused_step) < 2
    projection_path = (
        datastore.fiducial_projection_tiff_path() / "fiducial_max_projection.ome.tiff"
    )
    with TiffFile(projection_path) as tif:
        projection = tif.asarray()
        np.testing.assert_array_equal(
            projection, fused.read().result().squeeze().max(axis=0)
        )
        pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        assert float(pixels.attrib["PhysicalSizeY"]) == expected_spacing[1]
        assert float(pixels.attrib["PhysicalSizeX"]) == expected_spacing[2]

    masks = np.zeros(projection.shape, dtype=np.uint16)
    masks[1:5, 2:6] = 1
    datastore.save_global_cellpose_segmentation_image(
        masks, downsampling=np.asarray(expected_spacing) / np.asarray(native_spacing)
    )
    pixel_roi_path = datastore.cellpose_pixel_rois_path()
    pixel_roi_path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(
        pixel_roi_path,
        segment_fiducial.extract_pixel_rois(masks, np.array([1]), workers=1),
    )
    segment_fiducial.run_cellpose(
        tmp_path, outlines_only=True, roi_multiprocessing=False
    )
    np.testing.assert_array_equal(
        datastore.load_global_cellpose_segmentation_image(False), masks
    )
    scene = open_group(datastore.fiducial_projection_path()).validate(strict=True)
    product = scene.attrs["qi2lab"]["products"]["projection/fiducial"]
    projection_group = scene[product["path"]]
    labels = projection_group["labels"]
    label_name = labels.ome_metadata(strict=True).labels[0]
    mask_grid = labels[label_name].ome_metadata(strict=True).multiscales[0]
    projection_grid = projection_group.ome_metadata(strict=True).multiscales[0]
    assert mask_grid.datasets[0].transform == projection_grid.datasets[0].transform
    pixel_roi = roiread(pixel_roi_path)[0]
    global_roi = roiread(datastore.cellpose_global_rois_path())[0]
    xy = pixel_roi.coordinates()
    np.testing.assert_array_equal(xy.min(axis=0), [2, 1])
    np.testing.assert_array_equal(xy.max(axis=0), [5, 4])
    # This scene has an identity affine and starts at X=200, Y=100 microns.
    expected_xy = np.round(
        np.column_stack(
            (
                200 + xy[:, 0] * expected_spacing[2],
                100 + xy[:, 1] * expected_spacing[1],
            )
        ),
        2,
    )
    np.testing.assert_allclose(global_roi.coordinates(), expected_xy, atol=2e-5)

    # Re-exporting outlines must use exactly the same physical pixel scale.
    segment_fiducial.run_cellpose(
        tmp_path, outlines_only=True, roi_multiprocessing=False
    )
    regenerated = roiread(datastore.cellpose_global_rois_path())[0]
    np.testing.assert_array_equal(regenerated.coordinates(), global_roi.coordinates())


@pytest.mark.integration
@pytest.mark.parametrize("pre_registered", [False, True])
def test_single_tile_fusion_reopens_with_reflected_pixels_and_stage_origin(
    tmp_path, pre_registered
):
    from yaozarrs import open_group, v06

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.num_tiles = 1
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.experiment_order = np.array([[1, 1]])
    datastore.initialize_tile(0)
    datastore.voxel_size_zyx_um = [1, 1, 1]
    camera = np.diag([1, -1, 1, 1])
    datastore.save_local_stage_position_zyx_um([10, 20, 30], camera, tile=0, round=0)
    native = np.zeros((3, 10, 12), dtype=np.uint16)
    native[:, 2:-2, 2:-2] = np.arange(144).reshape(3, 6, 8) + 1
    datastore.save_local_corrected_image(native, tile=0, round=0)
    registration = registration_module.DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=registration_module.GlobalRegistrationConfig(
            fusion_workers=1
        ),
    )
    if pre_registered:
        registration.global_register(create_max_proj_tiff=False)

    registration.fuse_global_registered(create_max_proj_tiff=True)

    fiducial_scene = datastore.fused_scene_path()
    assert fiducial_scene == datastore.datastore_path / "fiducial_fusion.ome.zarr"
    assert isinstance(
        open_group(fiducial_scene).validate(strict=True).ome_metadata(), v06.Scene
    )
    assert datastore.fused_image_path().is_relative_to(fiducial_scene)
    assert (
        datastore.fiducial_projection_path()
        == datastore.datastore_path / "fiducial_projection.ome.zarr"
    )
    assert not (datastore.datastore_path / "fused.ome.zarr").exists()

    stored_affine, stored_origin, stored_spacing = (
        datastore.load_global_coord_xforms_um(tile=0)
    )
    np.testing.assert_array_equal(stored_affine, np.eye(4))
    np.testing.assert_array_equal(stored_origin, [10, 20, 30])
    np.testing.assert_array_equal(stored_spacing, [1, 1, 1])
    group = open_group(datastore.fused_image_path())
    dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0].path
    fused = group[dataset].to_tensorstore().read().result().squeeze()
    np.testing.assert_array_equal(fused, native[:, ::-1, :])
    np.testing.assert_array_equal(group.attrs["origin_zyx_um"], [10, -29, 30])
    with TiffFile(
        datastore.datastore_path
        / "fiducial_projection_tiff/fiducial_max_projection.ome.tiff"
    ) as tif:
        np.testing.assert_array_equal(tif.asarray(), native[:, ::-1, :].max(axis=0))
    assert datastore.datastore_state["Fused"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("camera_angle_deg", "register_rounds", "sharded"),
    [
        pytest.param(0, False, False, id="axis-aligned"),
        pytest.param(5, False, False, id="rotated-camera"),
        pytest.param(
            5, True, False, marks=pytest.mark.gpu, id="rotated-camera-multiround"
        ),
        pytest.param(
            5,
            True,
            True,
            marks=pytest.mark.gpu,
            id="sharded-rotated-camera-multiround",
        ),
        pytest.param(
            0,
            False,
            True,
            id="sharded-axis-aligned",
        ),
    ],
)
def test_two_tile_registration_fuses_directly_to_zarr(
    image_datastore, camera_angle_deg, register_rounds, sharded
):
    from scipy.ndimage import gaussian_filter, map_coordinates, shift
    from yaozarrs import open_group, v06

    volume = gaussian_filter(
        np.random.default_rng(41).random((54, 192, 288)), sigma=(2, 3, 3)
    )
    # Distinct fluorescent beads in the reference tile make the camera's
    # rotation observable in the fused pixels, independently of its metadata.
    beads = [(12, 80, 40), (24, 120, 50)]
    if register_rounds:
        beads.append((18, 100, 235))
    for z, y, x in beads:
        volume[z - 2 : z + 3, y - 2 : y + 3, x - 2 : x + 3] = 1
    angle = np.deg2rad(camera_angle_deg)
    camera = np.eye(4)
    camera[1:3, 1:3] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    for tile, (z_start, x_start, stage_x) in enumerate(
        [(0, 0, 30.0), (3, 96, 30.0 + 84 * 0.4)]
    ):
        image = (
            volume[z_start : z_start + 48, :, x_start : x_start + 192] * 60000
        ).astype(np.uint16)
        image[:, :4, :] = image[:, -4:, :] = 0
        image[:, :, :4] = image[:, :, -4:] = 0
        storage = {"chunks": (8, 64, 64), "shards": (48, 192, 192)} if sharded else {}
        image_datastore.save_local_corrected_image(image, tile=tile, round=0, **storage)
        if not register_rounds:
            image_datastore.save_local_deconvolved_fiducial_image(
                image, tile=tile, round=0, **storage
            )
        image_datastore.save_local_stage_position_zyx_um(
            [10, 20, stage_x], camera, tile=tile, round=0
        )
        if register_rounds:
            image_datastore.save_local_stage_position_zyx_um(
                [10, 20, stage_x], camera, tile=tile, round=1
            )
            # A later round has a measured bulk shift and a small smooth local
            # distortion; its sparse readout spots undergo the same motion.
            spots = np.zeros(image.shape, dtype=np.float32)
            for bead_z, bead_y, bead_x in beads:
                local_z, local_x = bead_z - z_start, bead_x - x_start
                if (
                    4 <= local_z < image.shape[0] - 4
                    and 4 <= local_x < image.shape[2] - 4
                ):
                    spots[local_z, bead_y, local_x] = 60000
            spots = gaussian_filter(spots, sigma=(1, 1, 1))
            z, y, x = np.indices(image.shape, dtype=np.float32)
            x += 1.5 * np.sin(2 * np.pi * y / image.shape[1])
            for reference, selection in ((image, {"round": 1}), (spots, {"bit": 1})):
                translated = shift(
                    reference.astype(np.float32),
                    shift=(1, 2, -2),
                    order=1,
                    mode="constant",
                    cval=0,
                )
                moving = (
                    map_coordinates(
                        translated, (z, y, x), order=1, mode="constant", cval=0
                    )
                    .clip(0, 65535)
                    .astype(np.uint16)
                )
                image_datastore.save_local_corrected_image(
                    moving, tile=tile, **selection, **storage
                )
    scene_group = open_group(image_datastore.tile_scene_path).validate(strict=True)
    scene = scene_group.ome_metadata(strict=True)
    assert isinstance(scene, v06.Scene)
    for tile in range(2):
        image_name = "corrected_data" if register_rounds else "decon_data"
        path = image_datastore.local_image_path(tile, image_name, round=0)
        assert isinstance(open_group(path).ome_metadata(strict=True), v06.Image)
        relative_path = str(path.relative_to(image_datastore.tile_scene_path))
        assert any(
            transform.input.path == relative_path
            and transform.output.name == f"tile{tile:04}/reference"
            for transform in scene.scene.coordinateTransformations
        )
    registration = registration_module.DataRegistration(
        image_datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=registration_module.GlobalRegistrationConfig(
            fusion_workers=1
        ),
    )

    if register_rounds:
        registration.register_all_tiles(process_readouts=False)
    registration.global_register(create_max_proj_tiff=True)

    affine, _, _ = image_datastore.load_global_coord_xforms_um(tile=1)
    np.testing.assert_allclose(affine[:3, 3], camera[:3, :3] @ [1.2, 0, 4.8], atol=0.2)
    group = open_group(image_datastore.fused_image_path()).validate(strict=True)
    model = group.ome_metadata(strict=True)
    assert isinstance(model, v06.Image)
    scene = (
        open_group(image_datastore.fused_scene_path())
        .validate(strict=True)
        .ome_metadata()
    )
    if register_rounds:
        from merfish3danalysis.utils.decode_warping import (
            warp_bit_image_to_reference,
            warp_image_to_reference_frame,
        )

        reopened = qi2labDataStore(image_datastore.datastore_path, validate=False)
        scene_group = open_group(reopened.tile_scene_path).validate(strict=True)
        transforms = {
            transform.name: transform
            for transform in scene_group.ome_metadata().scene.coordinateTransformations
        }
        statuses = scene_group.attrs["qi2lab"]["transform_status"]
        for tile, (z_start, x_start, _) in enumerate(
            [(0, 0, 30.0), (3, 96, 30.0 + 84 * 0.4)]
        ):
            tile_id = reopened.tile_ids[tile]
            stage = transforms[f"{tile_id}/round001/stage"]
            camera_transform = transforms[f"{tile_id}/round001/camera"]
            moving_stage = transforms[f"{tile_id}/round002/stage"]
            moving_camera = transforms[f"{tile_id}/round002/camera"]
            global_transform = transforms[f"{tile_id}/global"]
            decoder_spacing = transforms[f"{tile_id}/decoder_spacing"]
            decoder_origin = transforms[f"{tile_id}/decoder_origin"]
            reference_local = transforms[f"{tile_id}/round001/local"]
            local = transforms[f"{tile_id}/round002/local"]
            chromatic = transforms[f"{tile_id}/round002/bit002/chromatic"]
            deformation = transforms[f"{tile_id}/round002/deformable"]
            to_pixels = transforms[f"{tile_id}/round002/deformable/to_pixels"]
            to_physical = transforms[f"{tile_id}/round002/deformable/to_physical"]
            assert isinstance(deformation, v06.DisplacementsTransformation)
            assert isinstance(chromatic, v06.IdentityTransformation)
            assert chromatic.output == local.input
            assert stage.output == camera_transform.input
            assert camera_transform.output == global_transform.input
            assert moving_stage.input == local.input
            assert moving_stage.output == moving_camera.input
            assert decoder_spacing.output == decoder_origin.input
            assert decoder_origin.output == camera_transform.input
            assert stage.input == to_pixels.input
            assert to_pixels.output == deformation.input
            assert deformation.output == to_physical.input
            assert to_physical.output == local.output
            for name in (
                stage.name,
                camera_transform.name,
                moving_stage.name,
                moving_camera.name,
                global_transform.name,
                decoder_spacing.name,
                decoder_origin.name,
                reference_local.name,
                local.name,
                deformation.name,
            ):
                assert statuses[name] == "calculated"
            assert statuses[chromatic.name] == "pending"
            np.testing.assert_allclose(reference_local.affine, np.eye(4)[:3])
            np.testing.assert_allclose(moving_camera.affine, camera[:3])
            global_affine, global_origin, global_spacing = (
                reopened.load_global_coord_xforms_um(tile)
            )
            np.testing.assert_allclose(global_transform.affine, global_affine[:3])
            np.testing.assert_allclose(decoder_origin.translation, global_origin)
            np.testing.assert_allclose(decoder_spacing.scale, global_spacing)
            local_transform = reopened.load_local_round_transform_zyx_um(tile, 1)
            for actual, expected, tolerance in zip(
                local_transform[:3, 3], [0.4, 0.4, -0.8], [0.3, 0.2, 0.2], strict=True
            ):
                assert actual == pytest.approx(expected, abs=tolerance)
            np.testing.assert_allclose(local.affine, np.linalg.inv(local_transform)[:3])
            flow, flow_metadata = reopened.load_local_sofima_flow_field(
                tile=tile, round=1, return_future=False
            )
            assert flow.shape[0] == 3
            assert flow_metadata["valid_flow_vectors"] > 0
            fixed = reopened.load_local_corrected_image(
                tile=tile, round=0, return_future=False
            )
            moving = reopened.load_local_corrected_image(
                tile=tile, round=1, return_future=False
            )
            warp_options = {
                "transform_zyx_um": local_transform,
                "spacing_zyx_um": reopened.voxel_size_zyx_um,
                "reference_shape": fixed.shape,
            }
            affine_only = warp_image_to_reference_frame(moving, **warp_options)
            with_flow = warp_image_to_reference_frame(
                moving, loaded_flow_field=(flow, flow_metadata), **warp_options
            )
            # Ignore acquisition edges lost during the simulated displacement.
            interior = np.s_[6:-6, 10:-10, 10:-10]
            affine_error = np.mean((affine_only[interior] - fixed[interior]) ** 2)
            flow_error = np.mean((with_flow[interior] - fixed[interior]) ** 2)
            assert flow_error < 0.75 * affine_error, (tile_id, affine_error, flow_error)
            readout = reopened.load_local_corrected_image(
                tile=tile, bit=1, return_future=False
            )
            aligned = warp_bit_image_to_reference(
                readout,
                datastore=reopened,
                tile=tile,
                bit_id="bit002",
                emission_wavelength_um=0.67,
            )
            for bead_z, bead_y, bead_x in beads:
                local_z, local_x = bead_z - z_start, bead_x - x_start
                if (
                    4 <= local_z < aligned.shape[0] - 4
                    and 4 <= local_x < aligned.shape[2] - 4
                ):
                    neighborhood = aligned[
                        local_z - 3 : local_z + 4,
                        bead_y - 3 : bead_y + 4,
                        local_x - 3 : local_x + 4,
                    ]
                    peak = np.array(
                        np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
                    )
                    assert np.linalg.norm(peak - 3) <= 2
                    assert neighborhood.max() > 1000
    image_path = str(
        image_datastore.fused_image_path().relative_to(
            image_datastore.fused_scene_path()
        )
    )
    link = next(
        t for t in scene.scene.coordinateTransformations if t.input.path == image_path
    )
    assert isinstance(link, v06.ProjectAxisTransformation)
    assert link.droppedInputs == [0, 1]
    assert link.output.name == "world"
    datasets = model.multiscales[0].datasets
    assert len(datasets) == 2
    for dataset in datasets:
        codecs = group[dataset.path].metadata.model_dump(mode="json")["codecs"]
        blosc = next(codec for codec in codecs if codec["name"] == "blosc")
        assert blosc["configuration"]["cname"] == "zstd"
        assert blosc["configuration"]["clevel"] == 1
        assert blosc["configuration"]["shuffle"] == "bitshuffle"
    reopened = qi2labDataStore(image_datastore.datastore_path, validate=False)
    loaded = reopened.load_global_fiducial_image(return_future=False)
    assert loaded is not None
    fused, fused_affine, fused_origin, fused_spacing = loaded
    fused = fused.squeeze()
    np.testing.assert_array_equal(fused_affine, np.eye(4))
    np.testing.assert_allclose(fused_spacing, [0.4, 0.4, 0.4])
    assert fused.ndim == 3
    assert fused.shape[2] > 192
    assert fused.max() > 0
    if camera_angle_deg == 0:
        np.testing.assert_allclose(fused_origin, [10, 20, 30])
        expected = (volume[5:35, 20:160:2, 10:80] * 60000).astype(np.uint16)
        np.testing.assert_allclose(fused[5:35, 10:80, 10:80], expected, atol=2, rtol=0)
    else:
        reference_affine, _, _ = image_datastore.load_global_coord_xforms_um(tile=0)
        np.testing.assert_allclose(reference_affine, np.eye(4), atol=1e-6)
        for bead in beads:
            source_um = np.array([10, 20, 30]) + np.array(bead) * [0.4, 0.2, 0.4]
            expected_um = (camera @ np.r_[source_um, 1])[:3]
            index = np.rint((expected_um - fused_origin) / fused_spacing).astype(int)
            neighborhood = fused[
                index[0] - 2 : index[0] + 3,
                index[1] - 2 : index[1] + 3,
                index[2] - 2 : index[2] + 3,
            ]
            assert neighborhood.shape == (5, 5, 5)
            assert neighborhood.max() > 50000
    pyramid = group[datasets[1].path].to_tensorstore().read().result().squeeze()
    expected_pyramid = (
        fused[..., : pyramid.shape[-1] * 2]
        .reshape(*fused.shape[:-1], pyramid.shape[-1], 2)
        .mean(axis=-1)
        .astype(fused.dtype)
    )
    np.testing.assert_array_equal(pyramid, expected_pyramid)
    with TiffFile(
        image_datastore.datastore_path
        / "fiducial_projection_tiff/fiducial_max_projection.ome.tiff"
    ) as tif:
        np.testing.assert_array_equal(tif.asarray(), fused.max(axis=0))
    assert image_datastore.datastore_state["Fused"]


@pytest.mark.unit
@pytest.mark.parametrize("return_future", [False, True])
def test_single_tile_fiducial_publication_preserves_world_grid(
    image_datastore, return_future
):
    from tensorstore import WriteFutures
    from yaozarrs import v06

    datastore = image_datastore
    pixels = np.random.default_rng(7).poisson(100, (4, 12, 16)).astype(np.uint16)
    datastore.save_local_corrected_image(pixels, tile=0, round=0)
    fiducial = datastore.load_local_fiducial_image(tile=0, round=0, return_future=False)
    origin = [10, 20, 30]
    spacing = datastore.voxel_size_zyx_um
    completion = datastore.save_global_fiducial_image(
        fiducial, np.eye(4), origin, spacing, return_future=return_future
    )
    if return_future:
        assert isinstance(completion, WriteFutures)
        completion.result()
    else:
        assert completion is None

    path = datastore.fused_image_path()
    image = open_group(path).validate(strict=True).ome_metadata(strict=True)
    assert isinstance(image, v06.Image)
    scene_path = datastore.fused_scene_path()
    scene = open_group(scene_path).validate(strict=True).ome_metadata()
    relative = str(path.relative_to(scene_path))
    edge = next(
        t for t in scene.scene.coordinateTransformations if t.input.path == relative
    )
    assert edge.output.name == "world"
    assert isinstance(edge, v06.AffineTransformation)
    np.testing.assert_array_equal(edge.affine, np.eye(4)[:3])
    loaded, loaded_affine, loaded_origin, loaded_spacing = (
        datastore.load_global_fiducial_image(False)
    )
    np.testing.assert_array_equal(loaded, pixels)
    np.testing.assert_array_equal(loaded_affine, np.eye(4))
    np.testing.assert_array_equal(loaded_origin, origin)
    np.testing.assert_array_equal(loaded_spacing, spacing)
