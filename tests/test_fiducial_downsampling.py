"""Keep the fused segmentation grid, TIFF calibration, and ROI mapping aligned."""

from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pytest
import zarr
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from roifile import roiread
from tifffile import TiffFile

import merfish3danalysis.DataRegistration as registration_module
from merfish3danalysis.cli.qi2lab_microscopes import segment_fiducial
from merfish3danalysis.qi2labDataStore import qi2labDataStore


@pytest.mark.integration
@pytest.mark.parametrize(
    "native_spacing, expected_spacing",
    [
        ([0.315, 0.081, 0.081], [0.315, 0.316, 0.316]),
        ([0.32, 0.098, 0.112], [0.32, 0.323, 0.325]),
    ],
    ids=["simulation-nyquist", "anisotropic-xy"],
)
def test_fiducial_fusion_tiff_and_rois_share_downsampled_grid(
    tmp_path, monkeypatch, native_spacing, expected_spacing
):
    datastore_path = tmp_path / "qi2labdatastore"
    datastore = qi2labDataStore(datastore_path)
    datastore.voxel_size_zyx_um = native_spacing
    data = np.random.default_rng(8).integers(0, 65536, (7, 140, 168), dtype=np.uint16)
    source = zarr.create_array(tmp_path / "source.zarr", data=data, chunks=(3, 70, 84))
    sim = si_utils.get_sim_from_array(
        source,
        dims=("z", "y", "x"),
        scale=dict(zip("zyx", native_spacing, strict=True)),
        translation={"z": 12.0, "y": 100.0, "x": 200.0},
        transform_key="global_registered",
    )
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    options = registration_module._direct_zarr_fusion_kwargs
    monkeypatch.setattr(
        registration_module,
        "_direct_zarr_fusion_kwargs",
        lambda **kwargs: options(**kwargs, fusion_workers=1),
    )
    registration = registration_module.DataRegistration.__new__(
        registration_module.DataRegistration
    )
    registration._datastore = datastore
    registration._verbose = 0
    registration.fuse_global_fiducial_views(
        msims=[msim],
        create_max_proj_tiff=True,
    )

    fused_path = datastore._image_store_path(
        datastore._fused_root_path / "fused_fiducial_zyx"
    )
    attributes = datastore.load_image_metadata(fused_path)
    assert attributes["spacing_zyx_um"] == expected_spacing
    np.testing.assert_allclose(attributes["origin_zyx_um"], [12, 100, 200])
    np.testing.assert_array_equal(attributes["affine_zyx_um"], np.eye(4))
    fused = zarr.open_array(fused_path / "0", mode="r")
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
        np.testing.assert_array_equal(projection, fused[:].squeeze().max(axis=0))
        pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        assert float(pixels.attrib["PhysicalSizeY"]) == expected_spacing[1]
        assert float(pixels.attrib["PhysicalSizeX"]) == expected_spacing[2]

    masks = np.zeros(projection.shape, dtype=np.uint16)
    masks[1:5, 2:6] = 1
    model = SimpleNamespace(
        pretrained_model="test", eval=Mock(return_value=(masks, None, None))
    )
    monkeypatch.setattr(segment_fiducial, "qi2labDataStore", lambda _: datastore)
    monkeypatch.setattr(segment_fiducial.models, "CellposeModel", lambda **_: model)
    segment_fiducial.run_cellpose(tmp_path, use_gpu=False, roi_multiprocessing=False)
    np.testing.assert_array_equal(model.eval.call_args.args[0], projection[..., None])
    from yaozarrs import open_group

    mask_path = datastore_path / "segmentation/cellpose/masks_fiducial_iso_zyx.ome.zarr"
    mask_group = open_group(mask_path)
    np.testing.assert_array_equal(
        datastore.load_global_cellpose_segmentation_image(False), masks
    )
    np.testing.assert_allclose(
        mask_group.attrs["downsampling"], np.asarray(expected_spacing) / native_spacing
    )
    mask_grid = mask_group.attrs["ome"]["multiscales"][0]
    assert (
        mask_grid["datasets"][0]["coordinateTransformations"][0]["scale"]
        == expected_spacing[1:]
    )
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
    tmp_path, monkeypatch, pre_registered
):
    from yaozarrs import open_group

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.num_tiles = 1
    datastore.num_rounds = 1
    datastore.num_bits = 1
    datastore.voxel_size_zyx_um = [1, 1, 1]
    datastore._round_ids = ["round001"]
    datastore._bit_ids = ["bit001"]
    camera = np.diag([1, -1, 1, 1])
    datastore.local_image_path(0, "corrected_data", round=0).parent.mkdir(
        parents=True, exist_ok=True
    )
    datastore.save_local_stage_position_zyx_um([10, 20, 30], camera, tile=0, round=0)
    native = np.zeros((3, 10, 12), dtype=np.uint16)
    native[:, 2:-2, 2:-2] = np.arange(144).reshape(3, 6, 8) + 1
    datastore.save_local_corrected_image(native, tile=0, round=0)
    registration = registration_module.DataRegistration(
        datastore, decon_fiducial=False, verbose=0
    )
    options = registration_module._direct_zarr_fusion_kwargs
    monkeypatch.setattr(
        registration_module,
        "_direct_zarr_fusion_kwargs",
        lambda **kwargs: options(**kwargs, fusion_workers=1),
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
    fused = group["0"].to_zarr_python()[:].squeeze()
    np.testing.assert_array_equal(fused, native[:, ::-1, :])
    np.testing.assert_array_equal(group.attrs["origin_zyx_um"], [10, -29, 30])
    with TiffFile(
        datastore.datastore_path
        / "segmentation/cellpose/fiducial_max_projection.ome.tiff"
    ) as tif:
        np.testing.assert_array_equal(tif.asarray(), native[:, ::-1, :].max(axis=0))
    assert datastore.datastore_state["Fused"]
