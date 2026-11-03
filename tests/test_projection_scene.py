"""Projection and Cellpose labels share a separate NGFF 0.6 Scene."""

import numpy as np
import pytest
from tensorstore import TensorStore
from yaozarrs import open_group, v06


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
    from cellpose import models
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
    if not (models.MODEL_DIR / "cpsam_v2").is_file():
        pytest.skip("Cellpose cpsam_v2 model weights are not cached")

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
            [0, 0, stage_x_px * 0.107], np.eye(4), tile=tile, round=0
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
    projection, _affine, _origin, _spacing = reopened.load_global_fiducial_projection(
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
    roi_path = reopened.datastore_path / "cell_boundaries" / "cell_rois.zip"
    assert len(roiread(roi_path)) > 0
