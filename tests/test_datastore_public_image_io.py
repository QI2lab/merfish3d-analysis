"""Public image readers and writers must preserve their documented I/O modes."""

import json
from pathlib import Path

import numpy as np
import pytest
from tensorstore import Future, TensorStore, WriteFutures
from yaozarrs import open_group


@pytest.mark.unit
def test_decoded_codewords_and_normalization_remain_isolated_by_run(image_datastore):
    import pandas as pd

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    for run, gene in ((None, "default"), ("run1", "named")):
        frame = pd.DataFrame({"gene_id": [gene], "global_x": [1.0]})
        datastore.save_local_decoded_spots(frame, tile=0, decode_run_key=run)
        datastore.save_global_filtered_decoded_spots(frame, decode_run_key=run)
        datastore.save_decode_normalization_vectors(
            run, "global", np.ones(4), np.zeros(4), metadata={"run": gene}
        )

    root = datastore.datastore_path / "codewords"
    for path in (
        "tiles/tile0000.parquet",
        "tiles/runs/run1/tile0000.parquet",
        "global_filtered/decoded_features.parquet",
        "global_filtered/decoded_features.csv.gz",
        "global_filtered/runs/run1/decoded_features.parquet",
        "global_filtered/runs/run1/decoded_features.csv.gz",
        "normalization/default.json",
        "normalization/runs/run1.json",
    ):
        assert (root / path).is_file()

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    for run, gene in ((None, "default"), ("run1", "named")):
        assert reopened.load_local_decoded_spots(0, run).iloc[0]["gene_id"] == gene
        assert (
            reopened.load_global_filtered_decoded_spots(run).iloc[0]["gene_id"] == gene
        )
        assert reopened.load_decode_normalization_metadata(run, "global") == {
            "run": gene
        }
        normalization, background = reopened.load_decode_normalization_vectors(
            run, "global"
        )
        np.testing.assert_array_equal(normalization, np.ones(4))
        np.testing.assert_array_equal(background, np.zeros(4))


@pytest.mark.unit
def test_global_and_iterative_normalization_are_preserved_per_run(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    for run_index, run in enumerate((None, "run1", "run2")):
        for kind_index, kind in enumerate(("global", "iterative")):
            foreground = np.full(4, 1 + run_index * 2 + kind_index)
            background = foreground / 10
            image_datastore.save_decode_normalization_vectors(
                run, kind, foreground, background, metadata={"source": f"{run}-{kind}"}
            )

    reopened = qi2labDataStore(image_datastore.datastore_path)
    for run_index, run in enumerate((None, "run1", "run2")):
        for kind_index, kind in enumerate(("global", "iterative")):
            expected = np.full(4, 1 + run_index * 2 + kind_index)
            foreground, background = reopened.load_decode_normalization_vectors(
                run, kind
            )
            np.testing.assert_array_equal(foreground, expected)
            np.testing.assert_allclose(background, expected / 10)
            assert reopened.load_decode_normalization_metadata(run, kind) == {
                "source": f"{run}-{kind}"
            }


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["global", "iterative"])
def test_normalization_updates_are_visible_across_datastore_instances(
    image_datastore, kind
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    first = image_datastore
    first.save_decode_normalization_vectors(
        None, kind, np.ones(4), np.zeros(4), metadata={"revision": 1}
    )
    second = qi2labDataStore(first.datastore_path)
    second.save_decode_normalization_vectors(
        None, kind, np.full(4, 2), np.full(4, 0.5), metadata={"revision": 2}
    )

    foreground, background = first.load_decode_normalization_vectors(None, kind)
    np.testing.assert_array_equal(foreground, np.full(4, 2))
    np.testing.assert_array_equal(background, np.full(4, 0.5))
    foreground, background, metadata = first.load_decode_normalization_vectors(
        None, kind, include_metadata=True
    )
    np.testing.assert_array_equal(foreground, np.full(4, 2))
    np.testing.assert_array_equal(background, np.full(4, 0.5))
    assert metadata == {"revision": 2}
    np.testing.assert_array_equal(
        getattr(first, f"{kind}_normalization_vector"), foreground
    )
    np.testing.assert_array_equal(
        getattr(first, f"{kind}_background_vector"), background
    )

    setattr(second, f"{kind}_background_vector", np.full(4, 0.25))
    np.testing.assert_array_equal(
        getattr(first, f"{kind}_background_vector"), np.full(4, 0.25)
    )


@pytest.mark.integration
def test_concurrent_normalization_writers_preserve_both_updates(image_datastore):
    from concurrent.futures import ThreadPoolExecutor

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    first = image_datastore
    second = qi2labDataStore(first.datastore_path)
    first.save_decode_normalization_vectors(
        None, "iterative", np.ones(4), np.zeros(4), metadata={"source": "iterative"}
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        for value in range(1, 11):
            writes = (
                executor.submit(
                    first.save_decode_normalization_vectors,
                    None,
                    "global",
                    np.full(4, value),
                    np.full(4, value / 10),
                    metadata={"revision": value},
                ),
                executor.submit(
                    setattr, second, "iterative_background_vector", np.full(4, value)
                ),
            )
            for write in writes:
                write.result()

    reopened = qi2labDataStore(first.datastore_path)
    foreground, background, metadata = reopened.load_decode_normalization_vectors(
        None, "global", include_metadata=True
    )
    np.testing.assert_array_equal(foreground, np.full(4, 10))
    np.testing.assert_array_equal(background, np.ones(4))
    assert metadata == {"revision": 10}
    foreground, background, metadata = reopened.load_decode_normalization_vectors(
        None, "iterative", include_metadata=True
    )
    np.testing.assert_array_equal(foreground, np.ones(4))
    np.testing.assert_array_equal(background, np.full(4, 10))
    assert metadata == {"source": "iterative"}


@pytest.mark.unit
@pytest.mark.parametrize("run", [None, "named"])
@pytest.mark.parametrize("kind", ["global", "iterative"])
@pytest.mark.parametrize("metadata", [None, {"source": "test"}])
def test_decode_normalization_preserves_decode_mode_and_other_vectors(
    image_datastore, run, kind, metadata
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    other_kind = "iterative" if kind == "global" else "global"
    image_datastore.save_decode_normalization_vectors(
        run,
        other_kind,
        np.full(4, 3),
        np.full(4, 0.3),
        decode_mode="3d",
        metadata={"source": "previous"},
    )
    image_datastore.save_decode_normalization_vectors(
        run,
        kind,
        np.full(4, 2),
        np.full(4, 0.2),
        decode_mode="2d",
        metadata=metadata,
    )
    image_datastore.save_decode_normalization_vectors(
        run, kind, np.full(4, 2), np.full(4, 0.2)
    )

    reopened = qi2labDataStore(image_datastore.datastore_path)
    normalization, background = reopened.load_decode_normalization_vectors(run, kind)
    np.testing.assert_array_equal(normalization, np.full(4, 2))
    np.testing.assert_allclose(background, np.full(4, 0.2))
    other_normalization, other_background = reopened.load_decode_normalization_vectors(
        run, other_kind
    )
    np.testing.assert_array_equal(other_normalization, np.full(4, 3))
    np.testing.assert_allclose(other_background, np.full(4, 0.3))
    assert reopened.load_decode_normalization_metadata(run, kind) == metadata
    snapshot_normalization, snapshot_background, snapshot_metadata = (
        reopened.load_decode_normalization_vectors(run, kind, include_metadata=True)
    )
    np.testing.assert_array_equal(snapshot_normalization, normalization)
    np.testing.assert_array_equal(snapshot_background, background)
    assert snapshot_metadata == metadata
    assert reopened.load_decode_normalization_metadata(run, other_kind) == {
        "source": "previous"
    }

    root = reopened.datastore_path / "codewords" / "normalization"
    path = root / "default.json" if run is None else root / "runs" / f"{run}.json"
    assert json.loads(path.read_text())["decode_mode"] == "2d"


@pytest.mark.unit
@pytest.mark.parametrize("run", [".", "..", "../outside"])
def test_decoded_run_name_cannot_escape_its_product_folder(image_datastore, run):
    import pandas as pd

    with pytest.raises(ValueError, match="single run name"):
        image_datastore.save_local_decoded_spots(
            pd.DataFrame({"gene_id": ["gene"]}), tile=0, decode_run_key=run
        )


@pytest.mark.unit
@pytest.mark.parametrize("string_dtype", [None, "string"])
def test_validated_reopen_accepts_complete_named_decoding_run(
    image_datastore, string_dtype
):
    import pandas as pd

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    features = pd.DataFrame(
        {
            "gene_id": pd.Series(["gene", "gene2"], dtype=string_dtype),
            "global_x": [1.0, 2.0],
            "cell_id": pd.Series([1, pd.NA], dtype="Int64"),
        }
    )
    for tile in range(image_datastore.num_tiles):
        image_datastore.save_local_decoded_spots(features, tile, decode_run_key="run1")
    image_datastore.save_global_filtered_decoded_spots(features, decode_run_key="run1")
    image_datastore.datastore_state = {"DecodedSpots": True, "FilteredSpots": True}

    reopened = qi2labDataStore(image_datastore.datastore_path)
    pd.testing.assert_frame_equal(
        reopened.load_global_filtered_decoded_spots("run1"), features
    )


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
    assert (datastore.datastore_path / "camera/camera_noise.ome.zarr").is_dir()
    assert (datastore.datastore_path / "shading/shading_maps.ome.zarr").is_dir()
    assert (datastore.datastore_path / "psfs/images/psf_000.ome.zarr").is_dir()
    assert (datastore.datastore_path / "psfs/images/psf_001.ome.zarr").is_dir()
    assert (datastore.datastore_path / "psfs/psf_metadata/manifest.json").is_file()
    assert (
        datastore.datastore_path / "acquisition/codebook/codebook.parquet"
    ).is_file()
    assert (
        datastore.datastore_path
        / "acquisition/experiment_order/experiment_order.parquet"
    ).is_file()
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
@pytest.mark.parametrize("sharded", [False, True])
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
    image_datastore, writer, image_name, selection, expected_frame, shape, sharded
):
    from yaozarrs import v06

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], np.diag([1, -1, 1, 1]), tile=0, round=0
    )
    pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    options = (
        {"chunks": (1, 2, 2)[-len(shape) :], "shards": (2, 4, 4)[-len(shape) :]}
        if sharded
        else {}
    )
    completion = getattr(datastore, writer)(
        pixels, tile=0, return_future=sharded, **selection, **options
    )
    if sharded:
        assert isinstance(completion, WriteFutures)
        assert completion.result() is None

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    path = reopened.local_image_path(0, image_name, **selection)
    scene = open_group(reopened.tile_scene_path).validate(strict=True)
    image_group = open_group(path)
    image = image_group.ome_metadata(strict=True)
    assert isinstance(scene.ome_metadata(strict=True), v06.Scene)
    assert isinstance(image, v06.Image)
    relative_path = str(path.relative_to(reopened.tile_scene_path))
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
    if sharded:
        dataset = image.multiscales[0].datasets[0].path
        metadata = image_group[dataset].metadata.model_dump(mode="json")
        shard_codec = next(
            c for c in metadata["codecs"] if c["name"] == "sharding_indexed"
        )
        assert tuple(shard_codec["configuration"]["chunk_shape"]) == options["chunks"]
        compressor = next(
            c for c in shard_codec["configuration"]["codecs"] if c["name"] == "blosc"
        )
        assert compressor["configuration"]["cname"] == "zstd"
        assert compressor["configuration"]["clevel"] == 3
        assert compressor["configuration"]["shuffle"] == "shuffle"
        assert sum(p.is_file() for p in (path / dataset / "c").rglob("*")) == 1


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_image_metadata_preserves_corrected_and_derived_provenance(
    image_datastore, kind
):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    selection = {kind: 0}
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    datastore.save_local_corrected_image(
        pixels, tile=0, psf_idx=0, shading_correction=False, **selection
    )
    if kind == "round":
        datastore.save_local_deconvolved_fiducial_image(pixels, tile=0, round=0)
    else:
        datastore.save_local_deconvolved_readout_image(pixels, tile=0, bit=0)
        datastore.save_local_feature_predictor_image(pixels, tile=0, bit=0)
    datastore.save_local_corrected_image(
        pixels, tile=0, psf_idx=1, shading_correction=True, **selection
    )

    reopened = qi2labDataStore(datastore.datastore_path)
    corrected = reopened.load_image_metadata(
        reopened.local_image_path(0, "corrected_data", **selection)
    )
    deconvolved = reopened.load_image_metadata(
        reopened.local_image_path(0, "decon_data", **selection)
    )
    assert corrected["psf_idx"] == 1
    assert corrected["shading_correction"] is True
    assert "deconvolution" not in corrected
    assert deconvolved["psf_idx"] == 0
    assert deconvolved["shading_correction"] is False
    assert "deconvolution" not in deconvolved
    if kind == "bit":
        predictor = reopened.load_image_metadata(
            reopened.local_image_path(0, "feature_predictor_data", **selection)
        )
        assert predictor["psf_idx"] == 0
        assert predictor["shading_correction"] is False
        assert "deconvolution" not in predictor
    metadata = reopened.load_local_image_metadata(0, **selection)
    assert metadata["psf_idx"] == 1
    assert metadata["shading_correction"] is True
    assert "deconvolution" not in metadata
    assert (
        reopened.load_local_image_metadata(
            0, image_names=("corrected_data", "decon_data"), **selection
        )["psf_idx"]
        == 0
    )
    assert (
        reopened.load_local_image_metadata(
            0, image_names=("decon_data", "corrected_data"), **selection
        )["psf_idx"]
        == 1
    )


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_stale_deconvolution_is_not_selected_after_corrected_image_changes(
    image_datastore, kind
):
    from merfish3danalysis.DataRegistration import DataRegistration
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    selection = {kind: 0}
    corrected = np.ones((2, 3, 4), dtype=np.uint16)
    deconvolved = 2 * corrected
    datastore.save_local_corrected_image(corrected, tile=0, **selection)
    if kind == "round":
        datastore.save_local_deconvolved_fiducial_image(deconvolved, tile=0, round=0)
    else:
        import pandas as pd

        datastore.save_local_deconvolved_readout_image(deconvolved, tile=0, bit=0)
        datastore.save_local_feature_predictor_image(
            corrected, tile=0, bit=0, source_image="decon_data"
        )
        spots = pd.DataFrame({"z": [0.0], "y": [1.0], "x": [2.0]})
        datastore.save_local_feature_predictor_spots(spots, tile=0, bit=0)
    assert datastore.local_deconvolution_source_is_current(0, **selection) is True

    replacement = 9 * corrected
    datastore.save_local_corrected_image(replacement, tile=0, **selection)
    reopened = qi2labDataStore(datastore.datastore_path)
    assert reopened.local_deconvolution_source_is_current(0, **selection) is False
    registration = DataRegistration(reopened, num_gpus=0, verbose=0)
    if kind == "round":
        np.testing.assert_array_equal(
            reopened.load_local_deconvolved_fiducial_image(0, 0, False), deconvolved
        )
        np.testing.assert_array_equal(
            reopened.load_local_fiducial_image(0, 0, False), replacement
        )
        np.testing.assert_array_equal(
            reopened.load_local_fusion_image(0, round=0)[:], replacement
        )
        assert not registration._has_valid_deconvolved_fiducial_image(
            tile_id="tile0000", round_id="round001"
        )
    else:
        np.testing.assert_array_equal(
            reopened.load_local_deconvolved_readout_image(0, 0, False), deconvolved
        )
        assert not registration._has_valid_deconvolved_readout_image(
            tile_id="tile0000", bit_id="bit001"
        )
        with pytest.raises(ValueError, match="rerun readout preprocessing"):
            reopened.load_local_readout_image(0, 0, False)
        with pytest.raises(ValueError, match="rerun readout preprocessing"):
            reopened.load_local_fusion_image(0, bit=0)
        with pytest.raises(ValueError, match="rerun readout preprocessing"):
            reopened.load_local_feature_predictor_spots(0, 0)
        reopened.save_local_feature_predictor_image(
            replacement, tile=0, bit=0, source_image="corrected_data"
        )
        reopened.save_local_feature_predictor_spots(spots, tile=0, bit=0)
        pd.testing.assert_frame_equal(
            reopened.load_local_feature_predictor_spots(0, 0), spots
        )
        np.testing.assert_array_equal(
            reopened.load_local_readout_image(0, 0, False), replacement
        )
        np.testing.assert_array_equal(
            reopened.load_local_fusion_image(0, bit=0)[:], replacement
        )


@pytest.mark.unit
def test_stale_decon_without_predictor_falls_back_to_corrected(image_datastore):
    datastore = image_datastore
    corrected = np.ones((2, 3, 4), dtype=np.uint16)
    datastore.save_local_corrected_image(corrected, tile=0, bit=0)
    datastore.save_local_deconvolved_readout_image(2 * corrected, tile=0, bit=0)
    datastore.save_local_corrected_image(3 * corrected, tile=0, bit=0)

    np.testing.assert_array_equal(
        datastore.load_local_readout_image(0, 0, False), 3 * corrected
    )
    np.testing.assert_array_equal(
        datastore.load_local_fusion_image(0, bit=0)[:], 3 * corrected
    )


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["round", "bit"])
def test_untracked_deconvolution_remains_readable_but_does_not_satisfy_resume(
    image_datastore, kind
):
    from merfish3danalysis.DataRegistration import DataRegistration

    datastore = image_datastore
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    if kind == "round":
        datastore.save_local_deconvolved_fiducial_image(pixels, tile=0, round=0)
    else:
        datastore.save_local_deconvolved_readout_image(pixels, tile=0, bit=0)
    datastore.save_local_corrected_image(2 * pixels, tile=0, **{kind: 0})
    assert datastore.local_deconvolution_source_is_current(0, **{kind: 0}) is None
    registration = DataRegistration(datastore, num_gpus=0, verbose=0)
    if kind == "round":
        np.testing.assert_array_equal(
            datastore.load_local_fiducial_image(0, 0, False), pixels
        )
        assert not registration._has_valid_deconvolved_fiducial_image(
            tile_id="tile0000", round_id="round001"
        )
    else:
        np.testing.assert_array_equal(
            datastore.load_local_readout_image(0, 0, False), pixels
        )
        assert not registration._has_valid_deconvolved_readout_image(
            tile_id="tile0000", bit_id="bit001"
        )


@pytest.mark.unit
def test_round_linker_update_moves_the_readout_images_in_the_scene(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    datastore.save_local_corrected_image(pixels, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(pixels, tile=0, bit=0)
    datastore.save_local_round_linker(2, tile=0, bit=0)

    datastore = qi2labDataStore(datastore.datastore_path, validate=False)
    scene = open_group(datastore.tile_scene_path).validate(strict=True).ome_metadata()
    for image_name in ("corrected_data", "feature_predictor_data"):
        path = datastore.local_image_path(0, image_name, bit=0)
        relative_path = str(path.relative_to(datastore.tile_scene_path))
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
    scene = open_group(datastore.tile_scene_path).validate(strict=True).ome_metadata()
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
    group = open_group(reopened.tile_scene_path).validate(strict=True)
    scene = group.ome_metadata()
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    late = transforms["tile0001/round001/bit001/chromatic"]
    assert isinstance(late, v06.AffineTransformation)
    np.testing.assert_allclose(late.affine, affine[:3])
    changed = transforms["tile0000/round001/bit001/chromatic"]
    assert isinstance(changed, (v06.IdentityTransformation, v06.AffineTransformation))
    if isinstance(changed, v06.AffineTransformation):
        np.testing.assert_allclose(changed.affine, np.eye(4)[:3])

    # Every acquisition readout must retain the wavelength-specific correction
    # after the Scene is reopened, including tiles and rounds not changed above.
    for tile in reopened.tile_ids:
        for bit in reopened.bit_ids:
            round_index = reopened.load_local_round_linker(tile=tile, bit=bit) - 1
            round_id = reopened.round_ids[round_index]
            emission = reopened.load_local_wavelengths_um(tile=tile, bit=bit)[1]
            transform = transforms[f"{tile}/{round_id}/{bit}/chromatic"]
            assert transform.input.name == f"{tile}/{round_id}/{bit}/native"
            assert transform.output.name == f"{tile}/{round_id}/moving"
            assert group.attrs["qi2lab"]["transform_status"][transform.name] == (
                "calculated"
            )
            expected = affine if emission == 0.67 else np.eye(4)
            assert isinstance(
                transform, (v06.IdentityTransformation, v06.AffineTransformation)
            )
            actual = (
                np.eye(4)[:3]
                if isinstance(transform, v06.IdentityTransformation)
                else transform.affine
            )
            np.testing.assert_allclose(actual, expected[:3])


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
    scene = open_group(datastore.tile_scene_path).validate(strict=True).ome_metadata()
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

        scene = open_group(datastore.tile_scene_path).ome_metadata(strict=True)
        deformation = next(
            t
            for t in scene.scene.coordinateTransformations
            if t.name == "tile0000/round001/deformable"
        )
        assert isinstance(deformation, v06.DisplacementsTransformation)
        image_path = datastore.tile_scene_path / deformation.path
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
    before = (
        open_group(datastore.tile_scene_path).ome_metadata(strict=True).model_dump()
    )
    paths_before = set(datastore.datastore_path.rglob("zarr.json"))
    for key in selection:
        arguments = dict(selection, **{key: invalid})
        assert (
            getattr(datastore, writer)(np.zeros(shape), **arguments, **options) is None
        )
        assert getattr(datastore, reader)(**arguments) is None
    assert (
        open_group(datastore.tile_scene_path).ome_metadata(strict=True).model_dump()
        == before
    )
    assert set(datastore.datastore_path.rglob("zarr.json")) == paths_before


@pytest.mark.unit
@pytest.mark.parametrize("reference", [-1, 1, "round999", None, 0.5])
def test_sofima_reference_round_is_validated_before_writing(image_datastore, reference):
    from yaozarrs import open_group

    scene_path = image_datastore.tile_scene_path
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
def test_public_reader_rejects_corrupt_image_metadata(
    image_datastore,
    writer,
    reader,
    selection,
    options,
    relative_path,
    shape,
):
    datastore = image_datastore
    getattr(datastore, writer)(np.zeros(shape, dtype=np.uint16), **selection, **options)
    if writer in NATIVE_WRITERS:
        image_path = datastore.local_image_path(
            image_name=Path(relative_path).name.removesuffix(".ome.zarr"), **selection
        )
    else:
        image_path = datastore.fused_image_path()
    (image_path / "zarr.json").write_text('{"incomplete":')
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


@pytest.mark.integration
@pytest.mark.parametrize("source", ["corrected_data", "decon_data", None])
def test_predictor_source_selects_readout_for_decoding_and_fusion(
    image_datastore, source
):
    datastore = image_datastore
    corrected = np.full((3, 8, 10), 11, dtype=np.uint16)
    deconvolved = np.full_like(corrected, 23)
    datastore.save_local_corrected_image(corrected, tile=0, bit=0)
    datastore.save_local_deconvolved_readout_image(deconvolved, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        np.full(corrected.shape, 0.5, dtype=np.float32),
        tile=0,
        bit=0,
        source_image=source,
    )

    expected = corrected if source == "corrected_data" else deconvolved
    np.testing.assert_array_equal(
        datastore.load_local_readout_image(0, 0, return_future=False), expected
    )
    np.testing.assert_array_equal(
        datastore.load_local_fusion_image(0, bit=0)[:], expected
    )
    assert (
        datastore.load_local_image_metadata(
            0, bit=0, image_names=("feature_predictor_data",)
        ).get("source_image")
        == source
    )


@pytest.mark.unit
@pytest.mark.parametrize("source", ["corrected_data", "decon_data"])
def test_replacing_predictor_input_rejects_stale_prediction(image_datastore, source):
    datastore = image_datastore
    image = np.ones((3, 8, 10), dtype=np.uint16)
    datastore.save_local_corrected_image(image, tile=0, bit=0)
    datastore.save_local_deconvolved_readout_image(image * 2, tile=0, bit=0)
    datastore.save_local_feature_predictor_image(
        image, tile=0, bit=0, source_image=source
    )
    writer = (
        datastore.save_local_corrected_image
        if source == "corrected_data"
        else datastore.save_local_deconvolved_readout_image
    )
    writer(image * 3, tile=0, bit=0)

    with pytest.raises(ValueError, match="rerun readout preprocessing"):
        datastore.load_local_readout_image(0, 0, return_future=False)
    with pytest.raises(ValueError, match="rerun readout preprocessing"):
        datastore.load_local_fusion_image(0, bit=0)
    reader = (
        datastore.load_local_corrected_image
        if source == "corrected_data"
        else datastore.load_local_deconvolved_readout_image
    )
    np.testing.assert_array_equal(reader(0, bit=0, return_future=False), image * 3)
