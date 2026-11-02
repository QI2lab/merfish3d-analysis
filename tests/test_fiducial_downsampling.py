"""Keep the fused segmentation grid, TIFF calibration, and ROI mapping aligned."""

from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pytest
from roifile import roiread
from tifffile import TiffFile
from yaozarrs import open_group

import merfish3danalysis.DataRegistration as registration_module
from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.unit
@pytest.mark.parametrize(
    "native_spacing, expected_spacing",
    [
        ([0.315, 0.081, 0.081], [0.315, 0.316, 0.316]),
        ([0.32, 0.098, 0.112], [0.32, 0.323, 0.325]),
    ],
    ids=["simulation-nyquist", "anisotropic-xy"],
)
def test_fiducial_fusion_tiff_and_rois_share_downsampled_grid(
    tmp_path, image_datastore, monkeypatch, native_spacing, expected_spacing
):
    datastore_path = tmp_path / "qi2labdatastore"
    datastore = image_datastore
    datastore.voxel_size_zyx_um = native_spacing
    datastore.num_tiles = 1
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
        datastore_path / "segmentation/cellpose/fiducial_max_projection.ome.tiff"
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
    model = SimpleNamespace(
        pretrained_model="test", eval=Mock(return_value=(masks, None, None))
    )
    monkeypatch.setattr(segment_fiducial.models, "CellposeModel", lambda **_: model)
    segment_fiducial.run_cellpose(tmp_path, use_gpu=False, roi_multiprocessing=False)
    np.testing.assert_array_equal(model.eval.call_args.args[0], projection[..., None])
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
    roi_directory = projection_path.parent / "imagej_rois"
    pixel_roi = roiread(roi_directory / "pixel_spacing_rois.zip")[0]
    global_roi = roiread(roi_directory / "global_coords_rois.zip")[0]
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
    regenerated = roiread(roi_directory / "global_coords_rois.zip")[0]
    np.testing.assert_array_equal(regenerated.coordinates(), global_roi.coordinates())
    model.eval.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("pre_registered", [False, True])
def test_single_tile_fusion_reopens_with_reflected_pixels_and_stage_origin(
    tmp_path, pre_registered
):
    from yaozarrs import open_group

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
        / "segmentation/cellpose/fiducial_max_projection.ome.tiff"
    ) as tif:
        np.testing.assert_array_equal(tif.asarray(), native[:, ::-1, :].max(axis=0))
    assert datastore.datastore_state["Fused"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "camera_angle_deg", [0, 5], ids=["axis-aligned", "rotated-camera"]
)
def test_two_tile_registration_fuses_directly_to_zarr(
    image_datastore, camera_angle_deg
):
    from scipy.ndimage import gaussian_filter
    from yaozarrs import open_group, v06

    volume = gaussian_filter(
        np.random.default_rng(41).random((54, 192, 288)), sigma=(2, 3, 3)
    )
    # Distinct fluorescent beads in the reference tile make the camera's
    # rotation observable in the fused pixels, independently of its metadata.
    beads = [(12, 80, 40), (24, 120, 50)]
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
        image_datastore.save_local_corrected_image(image, tile=tile, round=0)
        image_datastore.save_local_deconvolved_fiducial_image(image, tile=tile, round=0)
        image_datastore.save_local_stage_position_zyx_um(
            [10, 20, stage_x], camera, tile=tile, round=0
        )
    scene_group = open_group(image_datastore.datastore_path).validate(strict=True)
    scene = scene_group.ome_metadata(strict=True)
    assert isinstance(scene, v06.Scene)
    for tile in range(2):
        path = image_datastore.local_image_path(tile, "decon_data", round=0)
        assert isinstance(open_group(path).ome_metadata(strict=True), v06.Image)
        relative_path = str(path.relative_to(image_datastore.datastore_path))
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

    registration.global_register(create_max_proj_tiff=True)

    affine, _, _ = image_datastore.load_global_coord_xforms_um(tile=1)
    np.testing.assert_allclose(affine[:3, 3], camera[:3, :3] @ [1.2, 0, 4.8], atol=0.2)
    group = open_group(image_datastore.fused_image_path()).validate(strict=True)
    model = group.ome_metadata(strict=True)
    assert isinstance(model, v06.Image)
    scene = (
        open_group(image_datastore.datastore_path).validate(strict=True).ome_metadata()
    )
    image_path = str(
        image_datastore.fused_image_path().relative_to(image_datastore.datastore_path)
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
        / "segmentation/cellpose/fiducial_max_projection.ome.tiff"
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
    scene = open_group(datastore.datastore_path).validate(strict=True).ome_metadata()
    relative = str(path.relative_to(datastore.datastore_path))
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
