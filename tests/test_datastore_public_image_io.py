"""Public image readers and writers must preserve their documented I/O modes."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from tensorstore import Future, TensorStore, WriteFutures
from yaozarrs import open_group

from merfish3danalysis.qi2labDataStore import qi2labDataStore

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
        "save_coord_of_xform_px",
        "load_coord_of_xform_px",
        {"tile": 0, "round": 0},
        {"block_size": [2, 4, 6], "block_stride": [1, 2, 3]},
        "fiducial/tile0000/round001/opticalflow_xform_px.ome.zarr",
        (3, 2, 3, 4),
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
        "segmentation/cellpose/masks_fiducial_iso_zyx.ome.zarr",
        (3, 4),
    ),
]


@pytest.fixture
def image_datastore(tmp_path):
    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.num_tiles = 1
    datastore.num_rounds = 1
    datastore.num_bits = 1
    # A minimal initialized acquisition, without unrelated codebook/PSF setup.
    datastore._round_ids = ["round001"]
    datastore._bit_ids = ["bit001"]
    datastore.voxel_size_zyx_um = [0.4, 0.2, 0.4]
    return datastore


@pytest.mark.integration
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
        if "flow" in relative_path
        or "xform" in relative_path
        or "feature_predictor" in relative_path
        else np.uint16
    )
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    completion = getattr(datastore, writer)(
        pixels, **selection, **options, return_future=return_future
    )
    if return_future:
        assert isinstance(completion, WriteFutures)
        assert completion.result() is None
    else:
        assert completion is None
    image_path = datastore.datastore_path / relative_path
    group = open_group(image_path)
    assert group.metadata.zarr_format == 3
    assert group.attrs["ome"]["version"] == "0.5"
    assert group["0"].metadata.shape == tuple(shape)
    for mode in [False, True, None]:
        loaded = getattr(datastore, reader)(**selection, return_future=mode)
        if reader == "load_coord_of_xform_px":
            assert len(loaded) == 3
            np.testing.assert_array_equal(loaded[1], options["block_size"])
            np.testing.assert_array_equal(loaded[2], options["block_stride"])
            loaded = loaded[0]
        elif reader == "load_local_sofima_flow_field":
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
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES
)
def test_public_writer_returns_write_handle_without_reading(
    mock_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
):
    datastore = mock_datastore
    completion = Mock()
    write = Mock(return_value=completion)
    read = Mock(side_effect=AssertionError("A write must not read pixels back"))
    monkeypatch.setattr(datastore, "_save_to_zarr_array", write)
    monkeypatch.setattr(datastore, "_load_from_zarr_array", read)
    monkeypatch.setattr(datastore, "_save_entity_attributes", Mock())
    pixels = np.zeros(shape, dtype=np.uint16)
    returned = getattr(datastore, writer)(
        pixels, **selection, **options, return_future=True
    )
    assert returned is completion
    completion.result.assert_called_once_with()
    assert Path(
        write.call_args.args[1]
    ) == datastore.datastore_path / relative_path.removesuffix(".ome.zarr")
    read.assert_not_called()


@pytest.mark.integration
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
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape, failure_stage",
    [
        (*case, stage)
        for case in IMAGE_CASES
        for stage in ["write", "completion", "metadata"]
        if stage != "metadata" or not case[0].startswith("save_global_")
    ],
)
def test_public_write_failures_propagate_without_updating_success_state(
    mock_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    failure_stage,
):
    datastore = mock_datastore
    original_state = dict(datastore.datastore_state)
    completion = Mock()
    write = Mock(return_value=completion)
    metadata = Mock()
    if failure_stage == "write":
        write.side_effect = OSError("disk full")
    elif failure_stage == "completion":
        completion.result.side_effect = OSError("disk full")
    else:
        metadata.side_effect = OSError("disk full")
    monkeypatch.setattr(datastore, "_save_to_zarr_array", write)
    monkeypatch.setattr(datastore, "_save_entity_attributes", metadata)
    with pytest.raises(OSError, match="disk full"):
        getattr(datastore, writer)(
            np.zeros(shape), **selection, **options, return_future=True
        )
    if failure_stage != "metadata":
        metadata.assert_not_called()
    assert datastore.datastore_state == original_state


@pytest.mark.unit
@pytest.mark.parametrize("invalid", [-1, 1, "unknown", 0.5, None])
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES[:7]
)
def test_invalid_local_image_identifiers_return_none_without_io(
    mock_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    invalid,
):
    write = Mock(side_effect=AssertionError("invalid selection must not write"))
    read = Mock(side_effect=AssertionError("invalid selection must not read"))
    monkeypatch.setattr(mock_datastore, "_save_to_zarr_array", write)
    monkeypatch.setattr(mock_datastore, "_load_from_zarr_array", read)
    for key in selection:
        arguments = dict(selection, **{key: invalid})
        assert (
            getattr(mock_datastore, writer)(np.zeros(shape), **arguments, **options)
            is None
        )
        assert getattr(mock_datastore, reader)(**arguments) is None
    write.assert_not_called()
    read.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES[:7]
)
def test_first_and_last_local_image_identifiers_select_expected_paths(
    mock_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    index,
    as_string,
):
    datastore = mock_datastore
    datastore.num_tiles = 2
    datastore.num_rounds = 2
    datastore.num_bits = 2
    datastore._round_ids = ["round001", "round002"]
    datastore._bit_ids = ["bit001", "bit002"]
    identifiers = {
        "tile": f"tile{index:04}",
        "round": f"round{index + 1:03}",
        "bit": f"bit{index + 1:03}",
    }
    arguments = {key: identifiers[key] if as_string else index for key in selection}
    write = Mock(return_value=None)
    monkeypatch.setattr(datastore, "_save_to_zarr_array", write)
    monkeypatch.setattr(datastore, "_save_entity_attributes", Mock())
    getattr(datastore, writer)(np.zeros(shape), **arguments, **options)
    path = Path(write.call_args.args[1])
    assert path.parent.parent.name == identifiers["tile"]
    assert path.parent.name == identifiers["round" if "round" in selection else "bit"]


@pytest.mark.unit
@pytest.mark.parametrize("reference", [-1, 1, "round999", None, 0.5])
def test_sofima_reference_round_is_validated_before_writing(
    mock_datastore, monkeypatch, reference
):
    write = Mock(side_effect=AssertionError("invalid reference must not write"))
    monkeypatch.setattr(mock_datastore, "_save_to_zarr_array", write)
    writer, _, selection, options, _, shape = IMAGE_CASES[6]
    options = dict(options, reference_round=reference)
    assert (
        getattr(mock_datastore, writer)(np.zeros(shape), **selection, **options) is None
    )
    write.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "writer, reader, selection, options, relative_path, shape", IMAGE_CASES
)
@pytest.mark.parametrize("error_type", [OSError, ValueError, AttributeError])
def test_public_reader_storage_errors_and_programming_errors(
    mock_datastore,
    monkeypatch,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
    error_type,
):
    datastore = mock_datastore
    monkeypatch.setattr(Path, "exists", Mock(return_value=True))
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
    mock_datastore,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
):
    assert getattr(mock_datastore, reader)(**selection, return_future=False) is None
