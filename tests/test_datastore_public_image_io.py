"""Public image readers and writers must preserve their documented I/O modes."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from tensorstore import Future, TensorStore, WriteFutures
from yaozarrs import open_group


@pytest.mark.unit
def test_calibration_images_written_before_tiles_reopen_with_channel_order(tmp_path):
    """Camera maps and per-channel PSFs retain their values and NGFF axes."""
    import pandas as pd
    from yaozarrs import v06

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.channels_in_data = ["fiducial", "readout"]
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.4]
    datastore.num_tiles = 1
    datastore.experiment_order = np.array([[1, 1]])
    datastore.tile_overlap = 0.1
    datastore.binning = 1
    datastore.e_per_ADU = 1.0
    datastore.na = 1.4
    datastore.ri = 1.5
    datastore.microscope_type = "synthetic"
    datastore.camera_model = "synthetic"
    datastore.codebook = pd.DataFrame({"gene_id": ["gene"], "bit01": [1]})

    noise = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    shading = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10 + 1
    psfs = [
        np.arange(27, dtype=np.float32).reshape(3, 3, 3),
        np.arange(45, dtype=np.float32).reshape(5, 3, 3) + 100,
    ]
    datastore.noise_map = noise
    datastore.channel_shading_maps = shading
    datastore.channel_psfs = psfs
    datastore.datastore_state = {"Calibrations": True}
    datastore.initialize_tile(0)
    datastore.save_local_corrected_image(
        np.ones((3, 3, 4), dtype=np.uint16), tile=0, bit=0, psf_idx=1
    )

    images = []
    for metadata_path in datastore.datastore_path.rglob("zarr.json"):
        try:
            group = open_group(metadata_path.parent)
            model = group.ome_metadata(strict=True)
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(model, v06.Image):
            multiscale = model.multiscales[0]
            array = group[multiscale.datasets[0].path].to_tensorstore()
            images.append((multiscale, array.read().result(), group.attrs))

    for expected, axes, psf_index in (
        (noise, "cyx", None),
        (shading, "cyx", None),
        (psfs[0], "zyx", 0),
        (psfs[1], "zyx", 1),
    ):
        matching = [
            (model, attrs)
            for model, actual, attrs in images
            if actual.dtype == expected.dtype and np.array_equal(actual, expected)
        ]
        assert len(matching) == 1
        model, attrs = matching[0]
        assert [axis.name for axis in model.axes] == list(axes)
        if axes == "cyx":
            assert attrs["channel_indices"] == [0, 1]
        else:
            assert attrs["psf_idx"] == psf_index

    reopened = qi2labDataStore(datastore.datastore_path)
    np.testing.assert_array_equal(reopened.noise_map, noise)
    np.testing.assert_array_equal(reopened.channel_shading_maps, shading)
    assert reopened.load_local_image_metadata(tile=0, bit=0)["psf_idx"] == 1
    assert len(reopened.channel_psfs) == len(psfs)
    for actual, expected in zip(reopened.channel_psfs, psfs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    unchecked = qi2labDataStore(datastore.datastore_path, validate=False)
    for actual, expected in zip(unchecked.channel_psfs, psfs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    reopened.channel_psfs = psfs[:1]
    reopened = qi2labDataStore(datastore.datastore_path)
    assert len(reopened.channel_psfs) == 1
    np.testing.assert_array_equal(reopened.channel_psfs[0], psfs[0])


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 4)])
@pytest.mark.parametrize(
    "writer, image_name, selection, expected_frame",
    [
        (
            "save_local_corrected_image",
            "corrected_data",
            {"round": 0},
            "tile0000/reference",
        ),
        (
            "save_local_corrected_image",
            "corrected_data",
            {"round": 1},
            "tile0000/round002/moving",
        ),
        (
            "save_local_corrected_image",
            "corrected_data",
            {"bit": 1},
            "tile0000/round002/bit002/native",
        ),
        (
            "save_local_deconvolved_fiducial_image",
            "decon_data",
            {"round": 1},
            "tile0000/round002/moving",
        ),
        (
            "save_local_deconvolved_readout_image",
            "decon_data",
            {"bit": 1},
            "tile0000/round002/bit002/native",
        ),
        (
            "save_local_feature_predictor_image",
            "feature_predictor_data",
            {"bit": 1},
            "tile0000/round002/bit002/native",
        ),
    ],
)
def test_native_images_are_published_in_the_acquisition_scene(
    image_datastore, writer, image_name, selection, expected_frame, shape
):
    from yaozarrs import v06

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], np.diag([1, -1, 1, 1]), tile=0, round=0
    )
    pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    getattr(datastore, writer)(pixels, tile=0, **selection)

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    path = reopened.local_image_path(0, image_name, **selection)
    scene = open_group(reopened.datastore_path).validate(strict=True)
    image = open_group(path).ome_metadata(strict=True)
    assert isinstance(scene.ome_metadata(strict=True), v06.Scene)
    assert isinstance(image, v06.Image)
    relative_path = str(path.relative_to(reopened.datastore_path))
    link = next(
        transform
        for transform in scene.ome_metadata().scene.coordinateTransformations
        if transform.input.path == relative_path
    )
    assert link.output.name == expected_frame
    if len(shape) == 2:
        assert isinstance(link, v06.ProjectAxisTransformation)
        assert link.createdOutputs == [0]
    else:
        assert isinstance(link, v06.IdentityTransformation)
    grid = image.multiscales[0].datasets[0].transform
    assert isinstance(grid, v06.ScaleTransformation)
    np.testing.assert_allclose(grid.scale, datastore.voxel_size_zyx_um[-len(shape) :])
    reader = {
        "save_local_corrected_image": reopened.load_local_corrected_image,
        "save_local_deconvolved_fiducial_image": reopened.load_local_deconvolved_fiducial_image,
        "save_local_deconvolved_readout_image": reopened.load_local_deconvolved_readout_image,
        "save_local_feature_predictor_image": reopened.load_local_feature_predictor_image,
    }[writer]
    lazy = reader(tile=0, return_future=None, **selection)
    assert isinstance(lazy, TensorStore)
    np.testing.assert_array_equal(lazy[..., 1:].read().result(), pixels[..., 1:])


@pytest.mark.unit
def test_round_linker_update_moves_the_readout_images_in_the_scene(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    datastore.save_local_corrected_image(pixels, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(pixels, tile=0, bit=0)
    datastore.save_local_round_linker(2, tile=0, bit=0)

    datastore = qi2labDataStore(datastore.datastore_path, validate=False)
    scene = open_group(datastore.datastore_path).validate(strict=True).ome_metadata()
    for image_name in ("corrected_data", "feature_predictor_data"):
        path = datastore.local_image_path(0, image_name, bit=0)
        relative_path = str(path.relative_to(datastore.datastore_path))
        link = next(
            transform
            for transform in scene.scene.coordinateTransformations
            if transform.input.path == relative_path
        )
        assert link.output.name == "tile0000/round002/bit001/native"
    assert datastore.load_local_round_linker(tile=0, bit=0) == 2
    chromatic = next(
        transform
        for transform in scene.scene.coordinateTransformations
        if transform.name == "tile0000/round002/bit001/chromatic"
    )
    assert chromatic.output.name == "tile0000/round002/moving"
    assert datastore.load_local_round_linker(tile=1, bit=0) == 1


@pytest.mark.unit
def test_chromatic_calibration_tracks_each_tile_and_late_wavelengths(image_datastore):
    from yaozarrs import v06

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    datastore.save_local_wavelengths_um((0.53, 0.58), tile=1, bit=0)
    datastore.save_local_round_linker(2, tile=0, bit=1)
    affine = np.eye(4)
    affine[1, 2] = 0.1
    affine[1, 3] = 0.25
    calibration = {
        "estimator": "decoded_rna_on_bit_weighted_centroid_yx_affine_graph",
        "reference_wavelength_um": 0.58,
        "channels": {
            "green": {"wavelength_um": 0.58, "affine_zyx_um": np.eye(4)},
            "red": {"wavelength_um": 0.67, "affine_zyx_um": affine},
        },
    }
    datastore.save_chromatic_affine_transforms_zyx_um(calibration)
    scene = open_group(datastore.datastore_path).validate(strict=True).ome_metadata()
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    for name in (
        "tile0000/round001/bit001/chromatic",
        "tile0000/round002/bit002/chromatic",
        "tile0001/round002/bit002/chromatic",
    ):
        assert isinstance(transforms[name], v06.AffineTransformation)
        np.testing.assert_allclose(transforms[name].affine, affine[:3])
    for name in (
        "tile0001/round001/bit001/chromatic",
        "tile0000/round001/bit003/chromatic",
    ):
        transform = transforms[name]
        assert isinstance(
            transform, (v06.IdentityTransformation, v06.AffineTransformation)
        )
        if isinstance(transform, v06.AffineTransformation):
            np.testing.assert_allclose(transform.affine, np.eye(4)[:3])

    # Per-tile emission can be published after calibration and can also change.
    datastore.save_local_wavelengths_um((0.63, 0.67), tile=1, bit=0)
    datastore.save_local_wavelengths_um((0.53, 0.58), tile=0, bit=0)
    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.load_local_wavelengths_um(tile=1, bit=0) == (0.63, 0.67)
    assert reopened.load_local_wavelengths_um(tile=0, bit=0) == (0.53, 0.58)
    np.testing.assert_allclose(
        reopened.load_chromatic_affine_transform_zyx_um(wavelength_um=0.67), affine
    )
    scene = open_group(reopened.datastore_path).validate(strict=True).ome_metadata()
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    late = transforms["tile0001/round001/bit001/chromatic"]
    assert isinstance(late, v06.AffineTransformation)
    np.testing.assert_allclose(late.affine, affine[:3])
    changed = transforms["tile0000/round001/bit001/chromatic"]
    assert isinstance(changed, (v06.IdentityTransformation, v06.AffineTransformation))
    if isinstance(changed, v06.AffineTransformation):
        np.testing.assert_allclose(changed.affine, np.eye(4)[:3])


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "malformed_wavelength",
        "rounded",
        "missing_first",
        "invalid_first",
        "non_optimizer",
    ],
)
def test_chromatic_scene_and_decoder_select_the_same_affine(image_datastore, case):
    from yaozarrs import v06

    from merfish3danalysis.utils.decode_warping import load_optimizer_chromatic_affine

    affine = np.eye(4)
    affine[1, 3] = 1 / 3
    channel = {"wavelength_um": 0.67, "affine_zyx_um": affine}
    channels = {"red": channel}
    estimator = "decoded_rna_on_bit_weighted_centroid_yx_affine_graph"
    if case == "malformed_wavelength":
        channels["other"] = {"wavelength_um": "invalid"}
    elif case == "missing_first":
        channels = {"missing": {"wavelength_um": 0.67}, "red": channel}
    elif case == "invalid_first":
        invalid = affine.copy()
        invalid[2, 2] = 1e-13
        channels = {
            "invalid": {"wavelength_um": 0.67, "affine_zyx_um": invalid},
            "red": channel,
        }
    elif case == "non_optimizer":
        estimator = "bead_calibration"

    datastore = image_datastore
    datastore.save_chromatic_affine_transforms_zyx_um(
        {"estimator": estimator, "channels": channels}
    )
    decoded = load_optimizer_chromatic_affine(datastore, wavelength_um=0.67)
    scene = open_group(datastore.datastore_path).validate(strict=True).ome_metadata()
    transform = next(
        t
        for t in scene.scene.coordinateTransformations
        if t.name == "tile0000/round001/bit001/chromatic"
    )
    if case in {"malformed_wavelength", "invalid_first", "non_optimizer"}:
        np.testing.assert_array_equal(decoded, np.eye(4))
        assert isinstance(transform, v06.IdentityTransformation)
    else:
        expected = np.asarray(affine, dtype=np.float32)
        np.testing.assert_array_equal(decoded, expected)
        assert isinstance(transform, v06.AffineTransformation)
        np.testing.assert_array_equal(transform.affine, expected[:3])


# Every public image writer, including both corrected-image destinations.
IMAGE_CASES = [
    (
        "save_local_corrected_image",
        "load_local_corrected_image",
        {"tile": 0, "round": 0},
        {},
        "fiducial/tile0000/round001/corrected_data.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_local_corrected_image",
        "load_local_corrected_image",
        {"tile": 0, "bit": 0},
        {},
        "readouts/tile0000/bit001/corrected_data.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_local_deconvolved_fiducial_image",
        "load_local_deconvolved_fiducial_image",
        {"tile": 0, "round": 0},
        {},
        "fiducial/tile0000/round001/decon_data.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_local_deconvolved_readout_image",
        "load_local_deconvolved_readout_image",
        {"tile": 0, "bit": 0},
        {},
        "readouts/tile0000/bit001/decon_data.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_local_feature_predictor_image",
        "load_local_feature_predictor_image",
        {"tile": 0, "bit": 0},
        {},
        "readouts/tile0000/bit001/feature_predictor_data.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_local_sofima_flow_field",
        "load_local_sofima_flow_field",
        {"tile": 0, "round": 0},
        {
            "reference_round": 0,
            "map_stride_zyx_px": [1, 2, 3],
            "map_box_start_xyz_px": [3, 2, 1],
            "map_box_size_xyz_px": [9, 4, 1],
            "reference_shape_zyx_px": [2, 3, 4],
            "moving_shape_zyx_px": [2, 3, 4],
        },
        "fiducial/tile0000/round001/local_sofima_flow_field.ome.zarr",
        (3, 2, 3, 4),
    ),
    (
        "save_global_fiducial_image",
        "load_global_fiducial_image",
        {},
        {
            "affine_zyx_um": np.eye(4),
            "origin_zyx_um": [2, 3, 4],
            "spacing_zyx_um": [0.4, 0.6, 0.8],
        },
        "fused/fused_fiducial_zyx.ome.zarr",
        (2, 3, 4),
    ),
    (
        "save_global_cellpose_segmentation_image",
        "load_global_cellpose_segmentation_image",
        {},
        {"downsampling": [1, 3, 2]},
        "projection/labels/cellpose",
        (3, 4),
    ),
]

NATIVE_WRITERS = {
    "save_local_corrected_image",
    "save_local_deconvolved_fiducial_image",
    "save_local_deconvolved_readout_image",
    "save_local_feature_predictor_image",
}


@pytest.mark.unit
@pytest.mark.parametrize("return_future", [False, True, None])
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES
)
def test_public_image_write_and_read_modes(
    image_datastore,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    return_future,
):
    datastore = image_datastore
    dtype = (
        np.float32
        if "flow" in relative_path or "feature_predictor" in relative_path
        else np.uint16
    )
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if writer == "save_global_cellpose_segmentation_image":
        datastore.save_global_fiducial_projection(
            np.zeros(shape, dtype=np.uint16),
            affine_zyx_um=np.eye(4),
            origin_zyx_um=[2, 3, 4],
            spacing_zyx_um=np.asarray(datastore.voxel_size_zyx_um)
            * options["downsampling"],
        )
    completion = getattr(datastore, writer)(
        pixels, **selection, **options, return_future=return_future
    )
    if return_future:
        assert isinstance(completion, WriteFutures)
        assert completion.result() is None
    else:
        assert completion is None
    if writer == "save_local_sofima_flow_field":
        from yaozarrs import v06

        scene = open_group(datastore.datastore_path).ome_metadata(strict=True)
        deformation = next(
            t
            for t in scene.scene.coordinateTransformations
            if t.name == "tile0000/round001/deformable"
        )
        assert isinstance(deformation, v06.DisplacementsTransformation)
        image_path = datastore.datastore_path / deformation.path
    elif writer in NATIVE_WRITERS:
        image_path = datastore.local_image_path(
            image_name=Path(relative_path).name.removesuffix(".ome.zarr"), **selection
        )
    elif writer == "save_global_fiducial_image":
        image_path = datastore.fused_image_path()
    elif writer == "save_global_cellpose_segmentation_image":
        from yaozarrs import v06

        scene = open_group(datastore.fiducial_projection_path()).validate(strict=True)
        projection_path = next(
            t.input.path
            for t in scene.ome_metadata().scene.coordinateTransformations
            if t.input.path is not None
        )
        labels = scene[projection_path]["labels"]
        labels_model = labels.ome_metadata(strict=True)
        assert isinstance(labels_model, v06.LabelsGroup)
        image_path = (
            datastore.fiducial_projection_path()
            / projection_path
            / "labels"
            / labels_model.labels[0]
        )
    else:
        image_path = datastore.datastore_path / relative_path
    group = open_group(image_path)
    assert group.metadata.zarr_format == 3
    metadata = group.ome_metadata(strict=True)
    assert metadata.version == "0.6"
    if writer == "save_global_cellpose_segmentation_image":
        assert isinstance(metadata, v06.LabelImage)
    assert group[metadata.multiscales[0].datasets[0].path].metadata.shape == tuple(
        shape
    )
    for mode in [False, True, None]:
        loaded = getattr(datastore, reader)(**selection, return_future=mode)
        if reader == "load_local_sofima_flow_field":
            assert len(loaded) == 2
            assert loaded[1]["map_stride_zyx_px"] == options["map_stride_zyx_px"]
            loaded = loaded[0]
        elif reader == "load_global_fiducial_image":
            assert len(loaded) == 4
            np.testing.assert_array_equal(loaded[1], options["affine_zyx_um"])
            np.testing.assert_array_equal(loaded[2], options["origin_zyx_um"])
            np.testing.assert_allclose(loaded[3], options["spacing_zyx_um"])
            loaded = loaded[0]
        if mode is True:
            assert isinstance(loaded, Future)
            actual = loaded.result()
        elif mode is None:
            assert isinstance(loaded, TensorStore)
            actual = loaded[1:].read().result()
        else:
            assert isinstance(loaded, np.ndarray)
            actual = loaded
        np.testing.assert_array_equal(actual, pixels[1:] if mode is None else pixels)


@pytest.mark.unit
@pytest.mark.parametrize("fiducial", [False, True])
@pytest.mark.parametrize("deconvolved", [False, True])
def test_preferred_image_reader_preserves_modes_with_corrected_fallback(
    image_datastore, fiducial, deconvolved
):
    datastore = image_datastore
    selection = {"tile": 0, "round" if fiducial else "bit": 0}
    corrected = np.ones((2, 3, 4), dtype=np.uint16)
    datastore.save_local_corrected_image(corrected, **selection)
    expected = corrected
    if deconvolved:
        writer = (
            datastore.save_local_deconvolved_fiducial_image
            if fiducial
            else datastore.save_local_deconvolved_readout_image
        )
        expected = corrected * 2
        writer(expected, **selection)
    reader = (
        datastore.load_local_fiducial_image
        if fiducial
        else datastore.load_local_readout_image
    )
    for mode in [False, True, None]:
        loaded = reader(**selection, return_future=mode)
        if mode is True:
            assert isinstance(loaded, Future)
            loaded = loaded.result()
        elif mode is None:
            assert isinstance(loaded, TensorStore)
            loaded = loaded.read().result()
        else:
            assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, expected)


@pytest.mark.unit
@pytest.mark.parametrize("invalid", [-1, 999, "unknown", 0.5, None])
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape",
    [case for case in IMAGE_CASES if "tile" in case[2]],
)
def test_invalid_local_image_identifiers_return_none_without_io(
    image_datastore,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    invalid,
):
    datastore = image_datastore
    before = open_group(datastore.datastore_path).ome_metadata(strict=True).model_dump()
    paths_before = set(datastore.datastore_path.rglob("zarr.json"))
    for key in selection:
        arguments = dict(selection, **{key: invalid})
        assert (
            getattr(datastore, writer)(np.zeros(shape), **arguments, **options) is None
        )
        assert getattr(datastore, reader)(**arguments) is None
    assert (
        open_group(datastore.datastore_path).ome_metadata(strict=True).model_dump()
        == before
    )
    assert set(datastore.datastore_path.rglob("zarr.json")) == paths_before


@pytest.mark.unit
@pytest.mark.parametrize("reference", [-1, 1, "round999", None, 0.5])
def test_sofima_reference_round_is_validated_before_writing(image_datastore, reference):
    from yaozarrs import open_group

    scene_path = image_datastore.datastore_path
    scene_before = open_group(scene_path).ome_metadata(strict=True).model_dump()
    assert (
        image_datastore.save_local_sofima_flow_field(
            np.zeros((3, 2, 3, 4), dtype=np.float32),
            tile=0,
            round=1,
            reference_round=reference,
            map_stride_zyx_px=(1, 1, 1),
            map_box_start_xyz_px=(0, 0, 0),
            map_box_size_xyz_px=(4, 3, 2),
            reference_shape_zyx_px=(2, 3, 4),
            moving_shape_zyx_px=(2, 3, 4),
        )
        is None
    )
    assert open_group(scene_path).ome_metadata(strict=True).model_dump() == scene_before


@pytest.mark.unit
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape",
    [
        case
        for case in IMAGE_CASES
        if case[0]
        not in {
            "save_local_sofima_flow_field",
            "save_global_cellpose_segmentation_image",
        }
    ],
)
@pytest.mark.parametrize("error_type", [OSError, ValueError, AttributeError])
def test_public_reader_storage_errors_and_programming_errors(
    image_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    error_type,
):
    datastore = image_datastore
    getattr(datastore, writer)(np.zeros(shape, dtype=np.uint16), **selection, **options)
    monkeypatch.setattr(
        datastore, "_load_from_zarr_array", Mock(side_effect=error_type("unreadable"))
    )
    if error_type is AttributeError:
        with pytest.raises(AttributeError, match="unreadable"):
            getattr(datastore, reader)(**selection, return_future=False)
    else:
        assert getattr(datastore, reader)(**selection, return_future=False) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES
)
def test_public_readers_return_none_for_absent_images(
    image_datastore,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
):
    assert getattr(image_datastore, reader)(**selection, return_future=False) is None
