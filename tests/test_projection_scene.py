"""Projection and Cellpose labels share a separate NGFF 0.6 Scene."""

import numpy as np
import pytest
from tensorstore import TensorStore
from yaozarrs import open_group, v06


@pytest.fixture
def segmented_projection(image_datastore):
    """Create a real projection, labels, and all Cellpose outline products."""
    from roifile import ImagejRoi, roiwrite

    datastore = image_datastore
    projection = np.arange(80, dtype=np.uint16).reshape(8, 10)
    datastore.save_global_fiducial_projection(
        projection, np.eye(4), [10, 20, 30], datastore.voxel_size_zyx_um
    )
    labels = np.zeros(projection.shape, dtype=np.uint16)
    labels[1:5, 2:7] = 1
    datastore.save_global_cellpose_segmentation_image(labels, [1, 1, 1])
    roi = ImagejRoi.frompoints([[2, 1], [6, 1], [6, 4], [2, 4]])
    for path in (
        datastore.cellpose_pixel_rois_path(),
        datastore.cellpose_global_rois_path(),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        roiwrite(path, [roi])
    outline = datastore.cellpose_outlines_path()
    outline.parent.mkdir(parents=True, exist_ok=True)
    outline.write_text('{"type":"FeatureCollection","features":[]}')
    datastore.datastore_state = {"SegmentedCells": True}
    return datastore


@pytest.mark.unit
@pytest.mark.parametrize("state_recorded", [False, True])
def test_replacing_projection_archives_existing_labels_and_outlines(
    segmented_projection, state_recorded
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = segmented_projection
    datastore.datastore_state = {"SegmentedCells": state_recorded}
    path = datastore.fiducial_projection_path()
    old_product = open_group(path).attrs["qi2lab"]["products"]["projection/fiducial"]
    old_revision = old_product["revision"]
    old_labels = datastore.load_global_cellpose_segmentation_image(False)
    files = (
        datastore.cellpose_pixel_rois_path(),
        datastore.cellpose_global_rois_path(),
        datastore.cellpose_outlines_path(),
    )
    original = [file.read_bytes() for file in files]
    datastore.save_global_fiducial_projection(
        np.full((8, 10), 42, dtype=np.uint16),
        np.eye(4),
        [10, 20, 30],
        datastore.voxel_size_zyx_um,
    )

    reopened = qi2labDataStore(datastore.datastore_path)
    assert not reopened.datastore_state["SegmentedCells"]
    assert reopened.load_global_cellpose_segmentation_image(False) is None
    boundaries = datastore.datastore_path / "cell_boundaries"
    for file, contents in zip(files, original, strict=True):
        assert not file.exists()
        assert (
            boundaries / "archive" / old_revision / file.relative_to(boundaries)
        ).read_bytes() == contents
    old_group = open_group(path)[old_product["path"]]["labels"]
    label = old_group[old_group.ome_metadata(strict=True).labels[0]]
    array = label.ome_metadata(strict=True).multiscales[0].datasets[0].path
    np.testing.assert_array_equal(
        label[array].to_tensorstore().read().result(), old_labels
    )


@pytest.mark.unit
def test_failed_projection_publication_restores_existing_segmentation(
    segmented_projection,
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = segmented_projection
    files = (
        datastore.cellpose_pixel_rois_path(),
        datastore.cellpose_global_rois_path(),
        datastore.cellpose_outlines_path(),
    )
    original = [file.read_bytes() for file in files]
    projection = datastore.load_global_fiducial_projection(False)[0]
    labels = datastore.load_global_cellpose_segmentation_image(False)

    invalid_affine = np.eye(4)
    invalid_affine[3, 3] = 0
    with pytest.raises(ValueError, match="homogeneous"):
        datastore.save_global_fiducial_projection(
            projection + 1, invalid_affine, [10, 20, 30], datastore.voxel_size_zyx_um
        )
    reopened = qi2labDataStore(datastore.datastore_path)
    assert reopened.datastore_state["SegmentedCells"]
    np.testing.assert_array_equal(
        reopened.load_global_fiducial_projection(False)[0], projection
    )
    np.testing.assert_array_equal(
        reopened.load_global_cellpose_segmentation_image(False), labels
    )
    assert [file.read_bytes() for file in files] == original


@pytest.mark.unit
@pytest.mark.parametrize("fail_after_archival", [False, True])
def test_failed_state_write_preserves_segmented_projection(
    segmented_projection, fail_after_archival
):
    import os
    import subprocess
    import sys

    if os.name != "posix":
        pytest.skip("File size limits require POSIX")
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = segmented_projection
    files = (
        datastore.cellpose_pixel_rois_path(),
        datastore.cellpose_global_rois_path(),
        datastore.cellpose_outlines_path(),
    )
    original = [file.read_bytes() for file in files]
    projection = datastore.load_global_fiducial_projection(False)[0]
    state_path = datastore.datastore_path / "datastore_state/state.json"
    state_bytes = state_path.read_bytes()
    revision = open_group(datastore.fiducial_projection_path()).attrs["qi2lab"][
        "products"
    ]["projection/fiducial"]["revision"]
    script = """
import errno
import resource
import sys
import numpy as np
from merfish3danalysis.qi2labDataStore import qi2labDataStore

datastore = qi2labDataStore(sys.argv[1])
_, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (int(sys.argv[2]), hard_limit))
try:
    datastore.save_global_fiducial_projection(
        np.ones((8, 10), dtype=np.uint16), np.eye(4), [10, 20, 30],
        datastore.voxel_size_zyx_um,
    )
except OSError as exc:
    assert exc.errno == errno.EFBIG
else:
    raise AssertionError("expected a file-size limit error")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(datastore.datastore_path),
            str(len(state_bytes) + 8 if fail_after_archival else 0),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert state_path.read_bytes() == state_bytes
    assert [file.read_bytes() for file in files] == original
    reopened = qi2labDataStore(datastore.datastore_path)
    assert reopened.datastore_state["SegmentedCells"]
    assert reopened.load_global_cellpose_segmentation_image(False) is not None
    np.testing.assert_array_equal(
        reopened.load_global_fiducial_projection(False)[0], projection
    )
    assert list(state_path.parent.iterdir()) == [state_path]
    if fail_after_archival:
        assert (
            datastore.datastore_path / "cell_boundaries/archive" / revision
        ).is_dir()


@pytest.mark.unit
def test_segmentation_archive_collision_preserves_both_copies(segmented_projection):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = segmented_projection
    products = open_group(datastore.fiducial_projection_path()).attrs["qi2lab"][
        "products"
    ]
    revision = products["projection/fiducial"]["revision"]
    pixel = datastore.cellpose_pixel_rois_path()
    original = pixel.read_bytes()
    collision = pixel.parent / "archive" / revision / pixel.name
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"existing archive")
    with pytest.raises(FileExistsError, match="archive already contains"):
        datastore.save_global_fiducial_projection(
            np.zeros((8, 10), dtype=np.uint16),
            np.eye(4),
            [10, 20, 30],
            datastore.voxel_size_zyx_um,
        )
    reopened = qi2labDataStore(datastore.datastore_path)
    assert reopened.datastore_state["SegmentedCells"]
    assert reopened.load_global_cellpose_segmentation_image(False) is not None
    assert pixel.read_bytes() == original
    assert collision.read_bytes() == b"existing archive"


@pytest.mark.integration
def test_competing_projection_publishers_preserve_segmentation_archive(
    segmented_projection,
):
    from concurrent.futures import ThreadPoolExecutor

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = segmented_projection
    scene = datastore.fiducial_projection_path()
    revision = open_group(scene).attrs["qi2lab"]["products"]["projection/fiducial"][
        "revision"
    ]
    paths = (
        datastore.cellpose_pixel_rois_path(),
        datastore.cellpose_global_rois_path(),
        datastore.cellpose_outlines_path(),
    )
    contents = [path.read_bytes() for path in paths]
    writers = [qi2labDataStore(datastore.datastore_path) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        completions = [
            executor.submit(
                writer.save_global_fiducial_projection,
                np.full((8, 10), value, dtype=np.uint16),
                np.eye(4),
                [10, 20, 30],
                writer.voxel_size_zyx_um,
            )
            for writer, value in zip(writers, (1, 2), strict=True)
        ]
        for completion in completions:
            completion.result()
    reopened = qi2labDataStore(datastore.datastore_path)
    assert not reopened.datastore_state["SegmentedCells"]
    assert reopened.load_global_cellpose_segmentation_image(False) is None
    boundaries = datastore.datastore_path / "cell_boundaries"
    for path, expected in zip(paths, contents, strict=True):
        assert not path.exists()
        assert (
            boundaries / "archive" / revision / path.relative_to(boundaries)
        ).read_bytes() == expected
    assert np.unique(reopened.load_global_fiducial_projection(False)[0]).item() in (
        1,
        2,
    )


@pytest.mark.unit
def test_replacing_projection_archives_standalone_rois_only(image_datastore):
    from roifile import ImagejRoi, roiwrite

    datastore = image_datastore
    image = np.zeros((8, 10), dtype=np.uint16)
    datastore.save_global_fiducial_projection(
        image, np.eye(4), [10, 20, 30], datastore.voxel_size_zyx_um
    )
    path = datastore.cellpose_global_rois_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    roiwrite(path, [ImagejRoi.frompoints([[0, 0], [3, 0], [3, 3]])])
    original = path.read_bytes()
    unrelated = path.parent / "notes.txt"
    unrelated.write_text("independent annotation")
    revision = open_group(datastore.fiducial_projection_path()).attrs["qi2lab"][
        "products"
    ]["projection/fiducial"]["revision"]
    datastore.save_global_fiducial_projection(
        image + 1, np.eye(4), [10, 20, 30], datastore.voxel_size_zyx_um
    )
    assert not path.exists()
    assert (path.parent / "archive" / revision / path.name).read_bytes() == original
    assert unrelated.read_text() == "independent annotation"


@pytest.mark.unit
def test_projection_scene_reopens_with_cellpose_labels(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    projection = np.arange(80, dtype=np.uint16).reshape(8, 10)
    masks = np.zeros(projection.shape, dtype=np.uint16)
    masks[1:5, 2:7] = 1
    masks[5:7, 6:9] = 2
    affine = np.eye(4, dtype=np.float32)
    origin = np.array([12, 21, 34], dtype=np.float32)
    spacing = np.array([0.4, 0.4, 0.4], dtype=np.float32)

    datastore.save_global_fiducial_projection(
        projection,
        affine_zyx_um=affine,
        origin_zyx_um=origin,
        spacing_zyx_um=spacing,
    )
    datastore.save_global_cellpose_segmentation_image(
        masks, downsampling=spacing / datastore.voxel_size_zyx_um
    )

    reopened = qi2labDataStore(datastore.datastore_path)
    loaded_projection, loaded_affine, loaded_origin, loaded_spacing = (
        reopened.load_global_fiducial_projection(return_future=False)
    )
    np.testing.assert_array_equal(loaded_projection, projection)
    np.testing.assert_allclose(loaded_affine, affine)
    np.testing.assert_allclose(loaded_origin, origin)
    np.testing.assert_allclose(loaded_spacing, spacing)
    lazy_masks = reopened.load_global_cellpose_segmentation_image(return_future=None)
    assert isinstance(lazy_masks, TensorStore)
    np.testing.assert_array_equal(lazy_masks.read().result(), masks)

    scene_path = reopened.fiducial_projection_path()
    assert scene_path == datastore.datastore_path / "fiducial_projection.ome.zarr"
    group = open_group(scene_path).validate(strict=True)
    scene = group.ome_metadata(strict=True)
    assert isinstance(scene, v06.Scene)
    image_paths = {
        ref.path
        for transform in scene.scene.coordinateTransformations
        for ref in (transform.input, transform.output)
        if ref.path is not None
    }
    projection_paths = [
        path
        for path in image_paths
        if isinstance(group[path].ome_metadata(strict=True), v06.Image)
        and not isinstance(group[path].ome_metadata(strict=True), v06.LabelImage)
    ]
    assert len(projection_paths) == 1
    image_path = projection_paths[0]
    image_group = group[image_path]
    image_model = image_group.ome_metadata(strict=True)
    assert [axis.name for axis in image_model.multiscales[0].axes] == ["y", "x"]
    labels_group = image_group["labels"]
    labels_model = labels_group.ome_metadata(strict=True)
    assert isinstance(labels_model, v06.LabelsGroup)
    assert len(labels_model.labels) == 1
    label_name = labels_model.labels[0]
    label_group = labels_group[label_name]
    label_model = label_group.ome_metadata(strict=True)
    assert isinstance(label_model, v06.LabelImage)
    assert [axis.name for axis in label_model.multiscales[0].axes] == ["y", "x"]
    projection_grid = image_model.multiscales[0].datasets[0].transform
    label_grid = label_model.multiscales[0].datasets[0].transform
    assert projection_grid.model_dump(exclude={"input", "output", "name"}) == (
        label_grid.model_dump(exclude={"input", "output", "name"})
    )
    source = label_model.image_label.source
    source_path = "../../" if source is None or source.image is None else source.image
    assert (
        scene_path / image_path / "labels" / label_name / source_path
    ).resolve() == (scene_path / image_path).resolve()
    dataset = label_model.multiscales[0].datasets[0]
    np.testing.assert_array_equal(
        label_group[dataset.path].to_tensorstore().read().result(), masks
    )
    replacement = masks.copy()
    replacement[0, 0] = 3
    completion = reopened.save_global_cellpose_segmentation_image(
        replacement,
        downsampling=spacing / datastore.voxel_size_zyx_um,
        return_future=True,
    )
    assert completion.result() is None
    np.testing.assert_array_equal(lazy_masks.read().result(), masks)
    np.testing.assert_array_equal(
        reopened.load_global_cellpose_segmentation_image(return_future=False),
        replacement,
    )
    group = open_group(scene_path).validate(strict=True)
    label_list = group[image_path]["labels"].ome_metadata(strict=True).labels
    assert len(label_list) == 1


@pytest.mark.integration
@pytest.mark.gpu
def test_real_cellpose_segments_fused_projection_scene(image_datastore):
    import torch
    from roifile import roiread
    from scipy.ndimage import gaussian_filter
    from skimage import data

    from merfish3danalysis.cli.qi2lab_microscopes.segment_fiducial import run_cellpose
    from merfish3danalysis.DataRegistration import (
        DataRegistration,
        GlobalRegistrationConfig,
    )
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    if not torch.cuda.is_available():
        pytest.skip("Cellpose integration requires CUDA")
    datastore = image_datastore
    datastore.voxel_size_zyx_um = [0.107, 0.107, 0.107]
    cell = data.cell().astype(np.float32)
    texture = gaussian_filter(
        np.random.default_rng(41).random((54, *cell.shape), dtype=np.float32),
        sigma=(1, 2, 2),
    )
    texture -= texture.min()
    texture /= texture.max()
    volume = (200 * cell[None, :, :] + 7000 * texture).astype(np.uint16)
    # Tile 1 starts 166 pixels into the specimen, but its measured stage
    # position is 12 pixels short. Its first plane is also shifted by 3.
    for tile, (z_start, x_start, stage_x_px) in enumerate([(0, 0, 0), (3, 166, 154)]):
        image = volume[z_start : z_start + 48, :, x_start : x_start + 384]
        datastore.save_local_corrected_image(image, tile=tile, round=0)
        datastore.save_local_stage_position_zyx_um(
            [10, 20, 30 + stage_x_px * 0.107], np.eye(4), tile=tile, round=0
        )
    registration = DataRegistration(
        datastore,
        decon_fiducial=False,
        verbose=0,
        global_registration_config=GlobalRegistrationConfig(
            registration_binning_zyx=(1, 2, 2), fusion_workers=1
        ),
    )
    registration.global_register(create_max_proj_tiff=True)
    affine, _origin, _spacing = datastore.load_global_coord_xforms_um(tile=1)
    np.testing.assert_allclose(affine[:3, 3], np.array([3, 0, 12]) * 0.107, atol=0.22)
    run_cellpose(datastore.datastore_path, use_gpu=True, roi_multiprocessing=False)

    reopened = qi2labDataStore(datastore.datastore_path)
    projection, affine, origin, spacing = reopened.load_global_fiducial_projection(
        return_future=False
    )
    masks = reopened.load_global_cellpose_segmentation_image(return_future=False)
    assert masks.shape == projection.shape
    assert np.count_nonzero(masks) > 0
    assert isinstance(
        open_group(reopened.fiducial_projection_path())
        .validate(strict=True)
        .ome_metadata(),
        v06.Scene,
    )
    pixel_rois = roiread(reopened.cellpose_pixel_rois_path())
    world_rois = roiread(reopened.cellpose_global_rois_path())
    assert len(pixel_rois) == len(world_rois) > 0
    np.testing.assert_allclose(affine, np.eye(4))
    assert np.all(np.abs(origin[1:]) > 1)
    for pixel_roi, world_roi in zip(pixel_rois, world_rois, strict=True):
        expected_xy = pixel_roi.coordinates() * spacing[[2, 1]] + origin[[2, 1]]
        np.testing.assert_allclose(
            world_roi.coordinates(), expected_xy, rtol=0, atol=0.011
        )

    projection_scene = reopened.fiducial_projection_path()
    old_product = open_group(projection_scene).attrs["qi2lab"]["products"][
        "projection/fiducial"
    ]
    pixel_path = reopened.cellpose_pixel_rois_path()
    global_path = reopened.cellpose_global_rois_path()
    original_pixel, original_global = pixel_path.read_bytes(), global_path.read_bytes()
    registration.fuse_global_registered(create_max_proj_tiff=False)

    reopened = qi2labDataStore(datastore.datastore_path)
    assert not reopened.datastore_state["SegmentedCells"]
    assert reopened.load_global_cellpose_segmentation_image(False) is None
    assert reopened.load_global_cellpose_roi_zip() is None
    assert not pixel_path.exists() and not global_path.exists()
    archive = (
        datastore.datastore_path / "cell_boundaries/archive" / old_product["revision"]
    )
    assert (archive / pixel_path.name).read_bytes() == original_pixel
    assert (archive / global_path.name).read_bytes() == original_global
    old_labels = open_group(projection_scene)[old_product["path"]]["labels"]
    label = old_labels[old_labels.ome_metadata(strict=True).labels[0]]
    dataset = label.ome_metadata(strict=True).multiscales[0].datasets[0].path
    np.testing.assert_array_equal(
        label[dataset].to_tensorstore().read().result(), masks
    )

    run_cellpose(datastore.datastore_path, use_gpu=True, roi_multiprocessing=False)
    reopened = qi2labDataStore(datastore.datastore_path)
    assert reopened.datastore_state["SegmentedCells"]
    regenerated_masks = reopened.load_global_cellpose_segmentation_image(False)
    assert regenerated_masks.shape == projection.shape
    assert np.any(regenerated_masks)
    assert len(roiread(reopened.cellpose_pixel_rois_path())) > 0
    assert len(roiread(reopened.cellpose_global_rois_path())) > 0
    assert (archive / pixel_path.name).read_bytes() == original_pixel
    assert (archive / global_path.name).read_bytes() == original_global
