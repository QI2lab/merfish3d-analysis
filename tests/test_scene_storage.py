"""Scene initialization for the NGFF 0.6 storage boundary."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest
from yaozarrs import open_group, v06

from merfish3danalysis.storage.ome import (
    ConcurrentSceneUpdateError,
    initialize_scene,
    publish_image,
    publish_transform,
    publish_transforms,
)


@pytest.mark.unit
def test_new_scene_is_valid_and_has_no_measured_position(tmp_path):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    assert datastore.tile_scene_path == datastore.datastore_path / "tiles.ome.zarr"
    assert not (datastore.datastore_path / "zarr.json").exists()
    group = open_group(datastore.tile_scene_path).validate(strict=True)
    scene = group.ome_metadata(strict=True)
    assert isinstance(scene, v06.Scene)
    assert scene.version == "0.6"
    assert len(scene.scene.coordinateSystems) == 1
    system = scene.scene.coordinateSystems[0]
    assert [axis.name for axis in system.axes] == ["z", "y", "x"]
    assert [axis.unit for axis in system.axes] == ["micrometer"] * 3
    assert len(scene.scene.coordinateTransformations) == 1
    transform = scene.scene.coordinateTransformations[0]
    assert isinstance(transform, v06.IdentityTransformation)
    assert transform.input.name == transform.output.name == system.name


@pytest.mark.unit
def test_previous_root_scene_requires_regenerating_datastore(tmp_path):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    path = tmp_path / "qi2labdatastore"
    initialize_scene(path)
    with pytest.raises(ValueError, match="regenerate"):
        qi2labDataStore(path)


@pytest.mark.unit
def test_scene_initialization_cannot_replace_existing_data(tmp_path):
    path = tmp_path / "qi2labdatastore"
    initialize_scene(path)
    marker = path / "existing-data"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        initialize_scene(path)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert isinstance(open_group(path).validate(strict=True).ome_metadata(), v06.Scene)


@pytest.mark.unit
def test_pending_scene_covers_every_tile_round_and_readout(image_datastore):
    datastore = image_datastore
    bits_by_round = {
        round_id: [datastore.bit_ids[int(bit) - 1] for bit in row[1:] if bit > 0]
        for round_id, row in zip(
            datastore.round_ids,
            datastore.experiment_order.to_numpy(),
            strict=True,
        )
    }
    group = open_group(datastore.tile_scene_path).validate(strict=True)
    reopened = group.ome_metadata(strict=True)

    transforms = {t.name: t for t in reopened.scene.coordinateTransformations}
    expected_names = {
        f"{tile}/{round_id}/{kind}"
        for tile in datastore.tile_ids
        for round_id in datastore.round_ids
        for kind in ("stage", "camera", "deformable", "local")
    }
    expected_names.update(f"{tile}/global" for tile in datastore.tile_ids)
    expected_names.update(
        f"{tile}/decoder_{kind}"
        for tile in datastore.tile_ids
        for kind in ("origin", "spacing")
    )
    expected_names.update(
        f"{tile}/{round_id}/{bit}/chromatic"
        for tile in datastore.tile_ids
        for round_id, bits in bits_by_round.items()
        for bit in bits
    )
    assert transforms.keys() == expected_names
    assert group.attrs["qi2lab"]["transform_status"] == dict.fromkeys(
        expected_names, "pending"
    )
    state = group.attrs["qi2lab"]
    for tile in datastore.tile_ids:
        for round_id, row in zip(
            datastore.round_ids, datastore.experiment_order.to_numpy(), strict=True
        ):
            assert state["entity_metadata"][f"fiducial/{tile}/{round_id}"][
                "bit_linker"
            ] == [int(value) for value in row[1:]]
            for bit in row[1:]:
                if bit:
                    assert state["round_links"][f"{tile}/bit{int(bit):03d}"] == round_id
    assert all(isinstance(t, v06.IdentityTransformation) for t in transforms.values())

    # The reference pixel-to-world chain adds stage before camera and global.
    stage = transforms["tile0000/round001/stage"]
    camera = transforms["tile0000/round001/camera"]
    global_transform = transforms["tile0000/global"]
    assert stage.output == camera.input
    assert camera.output == global_transform.input

    # Native readouts pass through chromatic and local registration toward round 0.
    deformation = transforms["tile0000/round002/deformable"]
    local = transforms["tile0000/round002/local"]
    chromatic = transforms["tile0000/round002/bit002/chromatic"]
    assert chromatic.output == local.input
    assert local.output == deformation.output


@pytest.mark.unit
def test_public_stage_write_updates_its_scene_transforms(image_datastore):
    camera = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    image_datastore.save_local_stage_position_zyx_um(
        [10, 20, 30], camera, tile=0, round=1
    )

    group = open_group(image_datastore.tile_scene_path).validate(strict=True)
    transforms = {
        t.name: t
        for t in group.ome_metadata(strict=True).scene.coordinateTransformations
    }
    stage = transforms["tile0000/round002/stage"]
    saved_camera = transforms["tile0000/round002/camera"]
    assert isinstance(stage, v06.TranslationTransformation)
    assert stage.translation == [10, 20, 30]
    assert isinstance(saved_camera, v06.AffineTransformation)
    np.testing.assert_array_equal(saved_camera.affine, camera[:3])
    statuses = group.attrs["qi2lab"]["transform_status"]
    assert statuses[stage.name] == statuses[saved_camera.name] == "calculated"
    assert statuses["tile0001/round002/stage"] == "pending"
    assert isinstance(transforms["tile0001/round002/stage"], v06.IdentityTransformation)


@pytest.mark.unit
def test_public_local_registration_updates_scene_transform(image_datastore):
    transform = np.array([[1, 0, 0, 2], [0, 0, -1, 3], [0, 1, 0, 4], [0, 0, 0, 1]])
    image_datastore.save_local_round_transform_zyx_um(transform, tile=0, round=1)

    group = open_group(image_datastore.tile_scene_path).validate(strict=True)
    transforms = {
        t.name: t
        for t in group.ome_metadata(strict=True).scene.coordinateTransformations
    }
    saved = transforms["tile0000/round002/local"]
    assert isinstance(saved, v06.AffineTransformation)
    forward = np.linalg.inv(transform)
    np.testing.assert_array_equal(saved.affine, forward[:3])
    moving_landmark = transform @ np.array([1, 2, 3, 1])
    np.testing.assert_array_equal(forward @ moving_landmark, [1, 2, 3, 1])
    np.testing.assert_array_equal(
        image_datastore.load_local_round_transform_zyx_um(tile=0, round=1), transform
    )
    statuses = group.attrs["qi2lab"]["transform_status"]
    assert statuses[saved.name] == "calculated"
    assert statuses["tile0001/round002/local"] == "pending"


@pytest.mark.unit
def test_conflicting_stage_batch_does_not_replace_committed_pair(image_datastore):
    path = image_datastore.tile_scene_path
    group = open_group(path)
    transforms = {
        t.name: t
        for t in group.ome_metadata(strict=True).scene.coordinateTransformations
    }
    names = ["tile0000/round001/stage", "tile0000/round001/camera"]
    original = [transforms[name] for name in names]
    replacement = [
        v06.TranslationTransformation(
            name=names[0],
            input=original[0].input,
            output=original[0].output,
            translation=[4, 5, 6],
        ),
        v06.AffineTransformation(
            name=names[1],
            input=original[1].input,
            output=original[1].output,
            affine=np.eye(4)[:3].tolist(),
        ),
    ]
    camera = np.diag([1, -1, 1, 1])
    image_datastore.save_local_stage_position_zyx_um([1, 2, 3], camera, tile=0, round=0)

    with pytest.raises(ConcurrentSceneUpdateError):
        publish_transforms(
            path,
            original,
            replacement,
            expected_statuses=dict.fromkeys(names, "pending"),
            expected_revisions=dict.fromkeys(names, 0),
        )

    reopened = open_group(path).validate(strict=True)
    statuses = reopened.attrs["qi2lab"]["transform_status"]
    assert statuses[names[0]] == statuses[names[1]] == "calculated"
    stored_stage, stored_camera = image_datastore.load_local_stage_position_zyx_um(
        tile=0, round=0
    )
    np.testing.assert_array_equal(stored_stage, [1, 2, 3])
    np.testing.assert_array_equal(stored_camera, camera)


@pytest.mark.unit
def test_disjoint_worker_transform_updates_survive_reopen(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    path = image_datastore.tile_scene_path
    group = open_group(path)
    before = {
        transform.name: transform
        for transform in group.ome_metadata(strict=True).scene.coordinateTransformations
    }
    alias = image_datastore.datastore_path.parent / "scene-alias"
    alias.symlink_to(image_datastore.datastore_path, target_is_directory=True)
    alias_datastore = qi2labDataStore(alias, validate=False)

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                datastore.save_local_stage_position_zyx_um,
                [tile + 1, -2, 3],
                np.eye(4),
                tile=tile,
                round=0,
            )
            for tile, datastore in enumerate((image_datastore, alias_datastore))
        ]
        for future in futures:
            future.result()

    group = open_group(path).validate(strict=True)
    after = {t.name: t for t in group.ome_metadata().scene.coordinateTransformations}
    names = [f"tile{tile:04}/round001/stage" for tile in range(2)]
    cameras = [f"tile{tile:04}/round001/camera" for tile in range(2)]
    for tile, name in enumerate(names):
        assert isinstance(after[name], v06.TranslationTransformation)
        assert after[name].translation == [tile + 1, -2, 3]
    assert all(
        after[name] == before[name] for name in before.keys() - set(names + cameras)
    )
    statuses = group.attrs["qi2lab"]["transform_status"]
    assert all(statuses[name] == "calculated" for name in names + cameras)
    assert statuses["tile0000/round002/local"] == "pending"
    reopened = qi2labDataStore(image_datastore.datastore_path, validate=False)
    for tile in range(2):
        stage, camera = reopened.load_local_stage_position_zyx_um(tile=tile, round=0)
        np.testing.assert_array_equal(stage, [tile + 1, -2, 3])
        np.testing.assert_array_equal(camera, np.eye(4))


@pytest.mark.unit
def test_concurrent_tile_metadata_updates_survive_reopen(image_datastore):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(image_datastore.save_local_bit_linker, [0, 2], tile=0, round=1),
            pool.submit(
                image_datastore.save_local_wavelengths_um,
                (0.53, 0.61),
                tile=0,
                round=1,
            ),
        ]
        for future in futures:
            future.result()

    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    image_datastore.save_local_corrected_image(pixels, tile=0, round=1)
    reopened = qi2labDataStore(image_datastore.datastore_path, validate=False)
    assert reopened.load_local_bit_linker(tile=0, round=1) == [0, 2]
    np.testing.assert_allclose(
        reopened.load_local_wavelengths_um(tile=0, round=1), (0.53, 0.61)
    )
    assert reopened.load_local_round_linker(tile=0, bit=1) == 2
    np.testing.assert_array_equal(
        reopened.load_local_corrected_image(tile=0, round=1, return_future=False),
        pixels,
    )
    assert not (reopened.tile_scene_path / "fiducial").exists()
    assert not (reopened.tile_scene_path / "readouts").exists()
    open_group(reopened.tile_scene_path).validate(strict=True)


@pytest.mark.unit
@pytest.mark.parametrize("image_datastore", ["zero_slot"], indirect=True)
def test_scene_preserves_nominal_channel_slots_and_linker_overrides(image_datastore):
    import pandas as pd

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    original = datastore.experiment_order.copy()
    changed = original.copy()
    changed.iloc[0, 1:] = [0, 1]
    path = (
        datastore.datastore_path
        / "acquisition/experiment_order/experiment_order.parquet"
    )
    original_parquet = path.read_bytes()
    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    datastore.save_local_corrected_image(pixels, tile=0, round=0)
    datastore.save_local_bit_linker([0, 1], tile=0, round=0)

    datastore.experiment_order = original.copy()
    with pytest.raises(ValueError, match="Experiment order"):
        datastore.experiment_order = changed

    assert path.read_bytes() == original_parquet
    pd.testing.assert_frame_equal(datastore.experiment_order, original)
    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.load_local_bit_linker(tile=0, round=0) == [0, 1]
    np.testing.assert_array_equal(
        reopened.load_local_corrected_image(tile=0, round=0, return_future=False),
        pixels,
    )
    assert open_group(reopened.tile_scene_path).attrs["qi2lab"]["experiment_order"] == (
        original.to_numpy().tolist()
    )

    changed.to_parquet(path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="Experiment order"):
        qi2labDataStore(datastore.datastore_path, validate=False)


@pytest.mark.unit
def test_scene_preserves_acquisition_channel_order(image_datastore):
    import json

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = image_datastore
    channels = list(datastore.channels_in_data)
    changed = [channels[0], channels[2], channels[1]]
    path = (
        datastore.datastore_path / "acquisition/acquisition_metadata/acquisition.json"
    )
    datastore.channels_in_data = channels.copy()
    original_metadata = path.read_bytes()

    with pytest.raises(ValueError, match="Acquisition channels"):
        datastore.channels_in_data = changed
    assert path.read_bytes() == original_metadata
    assert datastore.channels_in_data == channels

    reopened = qi2labDataStore(datastore.datastore_path, validate=False)
    assert reopened.channels_in_data == channels
    assert (
        open_group(reopened.tile_scene_path).attrs["qi2lab"]["channels_in_data"]
        == channels
    )

    metadata = json.loads(original_metadata)
    metadata["channels_in_data"] = changed
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Acquisition channels"):
        qi2labDataStore(datastore.datastore_path, validate=False)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["num_rounds", "num_tiles", "num_bits"])
def test_scene_preserves_acquisition_counts(image_datastore, name):
    datastore = image_datastore
    count = getattr(datastore, name)
    path = (
        datastore.datastore_path / "acquisition/acquisition_metadata/acquisition.json"
    )
    setattr(datastore, name, count)
    original_metadata = path.read_bytes()
    with pytest.raises(ValueError, match="does not match the tile Scene"):
        setattr(datastore, name, count + 1)
    assert getattr(datastore, name) == count
    assert path.read_bytes() == original_metadata
    if name == "num_bits":
        from merfish3danalysis.qi2labDataStore import qi2labDataStore

        reopened = qi2labDataStore(datastore.datastore_path)
        assert reopened.num_bits == count
        assert reopened.bit_ids == datastore.bit_ids


@pytest.mark.unit
def test_setting_bit_count_before_scene_updates_bit_ids(tmp_path):
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.num_bits = 2
    assert datastore.bit_ids == ["bit001", "bit002"]
    datastore.num_bits = 3
    assert datastore.bit_ids == ["bit001", "bit002", "bit003"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_order, mismatch",
    [
        ([[1, 1, 3], [2, 2, 4], [3, 5, 6]], "rounds"),
        ([[1, 1, 3], [2, 2, 5]], "bits"),
    ],
)
def test_rejected_experiment_order_preserves_acquisition(
    tmp_path, invalid_order, mismatch
):
    import pandas as pd

    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(tmp_path / "qi2labdatastore")
    datastore.channels_in_data = ["fiducial", "readout1", "readout2"]
    datastore.experiment_order = np.array([[1, 1, 3], [2, 2, 4]])
    original = datastore.experiment_order.copy()
    path = (
        datastore.datastore_path
        / "acquisition/experiment_order/experiment_order.parquet"
    )
    original_bytes = path.read_bytes()

    with pytest.raises(AssertionError, match=f"Number of {mismatch}"):
        datastore.experiment_order = np.asarray(invalid_order)

    pd.testing.assert_frame_equal(datastore.experiment_order, original)
    assert path.read_bytes() == original_bytes
    assert datastore.num_rounds == 2
    assert datastore.num_bits == 4
    assert datastore.round_ids == ["round001", "round002"]
    assert datastore.bit_ids == ["bit001", "bit002", "bit003", "bit004"]


@pytest.mark.unit
def test_stale_update_does_not_replace_a_completed_transform(pending_scene):
    scene = open_group(pending_scene).ome_metadata(strict=True)
    original = next(
        transform
        for transform in scene.scene.coordinateTransformations
        if transform.name == "tile0000/round002/local"
    )
    first = v06.AffineTransformation(
        name=original.name,
        input=original.input,
        output=original.output,
        affine=[[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]],
    )
    second = first.model_copy(
        update={"affine": [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6]]}
    )
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(publish_transform, pending_scene, original, replacement)
            for replacement in (first, second)
        ]
        committed = []
        conflicts = 0
        for replacement, future in zip((first, second), futures, strict=True):
            try:
                future.result()
                committed.append(replacement)
            except ConcurrentSceneUpdateError:
                conflicts += 1
    assert len(committed) == conflicts == 1

    group = open_group(pending_scene).validate(strict=True)
    observed = next(
        transform
        for transform in group.ome_metadata().scene.coordinateTransformations
        if transform.name == original.name
    )
    assert observed == committed[0]
    assert group.attrs["qi2lab"]["transform_status"][original.name] == "calculated"


@pytest.mark.unit
def test_calculated_identity_is_distinct_from_pending_identity(image_datastore):
    path = image_datastore.tile_scene_path
    scene = open_group(path).ome_metadata(strict=True)
    original = next(
        transform
        for transform in scene.scene.coordinateTransformations
        if transform.name == "tile0000/round001/local"
    )
    image_datastore.save_local_round_transform_zyx_um(np.eye(4), tile=0, round=0)
    with pytest.raises(ConcurrentSceneUpdateError):
        publish_transform(path, original, original)
    calculated = next(
        transform
        for transform in open_group(path).ome_metadata().scene.coordinateTransformations
        if transform.name == original.name
    )
    assert isinstance(calculated, v06.AffineTransformation)
    np.testing.assert_array_equal(calculated.affine, np.eye(4)[:3])
    image_datastore.save_local_round_transform_zyx_um(np.eye(4), tile=0, round=0)
    with pytest.raises(ConcurrentSceneUpdateError):
        publish_transform(
            path,
            calculated,
            calculated,
            expected_status="calculated",
            expected_revision=1,
        )

    reopened = open_group(path).validate(strict=True)
    assert reopened.attrs["qi2lab"]["transform_status"][original.name] == "calculated"
    assert (
        reopened.attrs["qi2lab"]["transform_status"]["tile0001/round001/local"]
        == "pending"
    )


@pytest.mark.unit
def test_transform_publication_preserves_existing_image_pixels(image_datastore):
    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    image_datastore.save_local_corrected_image(pixels, tile=0, round=0)
    path = image_datastore.tile_scene_path
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    image_datastore.save_local_round_transform_zyx_um(transform, tile=0, round=1)

    open_group(path).validate(strict=True)
    actual = image_datastore.load_local_corrected_image(
        tile=0, round=0, return_future=False
    )
    np.testing.assert_array_equal(actual, pixels)


@pytest.mark.unit
def test_published_image_supports_native_reads_reopen_and_replacement(image_datastore):
    pixels = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    image_datastore.save_local_corrected_image(pixels, tile=0, bit=1)

    reopened = image_datastore.load_local_corrected_image(
        tile=0, bit=1, return_future=None
    )
    np.testing.assert_array_equal(reopened.read().result(), pixels)
    np.testing.assert_array_equal(
        reopened[:, 1:3, 2:].read().result(), pixels[:, 1:3, 2:]
    )
    image_datastore.save_local_corrected_image(pixels + 1, tile=0, bit=1)
    np.testing.assert_array_equal(
        image_datastore.load_local_corrected_image(tile=0, bit=1, return_future=False),
        pixels + 1,
    )
    # A reader opened before publication retains its immutable image revision.
    np.testing.assert_array_equal(reopened.read().result(), pixels)
    open_group(image_datastore.tile_scene_path).validate(strict=True)


@pytest.mark.unit
def test_concurrent_image_writers_cannot_replace_each_other(image_datastore):
    """Test publication conflicts using the same revision in both workers."""
    pixels = np.ones((2, 3, 4), dtype=np.uint16)
    image_datastore.save_local_corrected_image(pixels * 3, tile=1, bit=0)
    image = open_group(
        image_datastore.local_image_path(1, "corrected_data", bit=0)
    ).ome_metadata(strict=True)
    selection = "tile0001/bit001/corrected_data"
    path = image_datastore.tile_scene_path
    product = open_group(path).attrs["qi2lab"]["products"][selection]
    frame, revision = product["frame"], product["revision"]
    candidates = (pixels, pixels * 2)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(
                publish_image,
                path,
                selection,
                frame,
                image,
                data,
                expected_revision=revision,
            )
            for data in candidates
        ]
        committed = []
        conflicts = 0
        for data, future in zip(candidates, futures, strict=True):
            try:
                future.result()
                committed.append(data)
            except ConcurrentSceneUpdateError:
                conflicts += 1
    assert len(committed) == conflicts == 1
    np.testing.assert_array_equal(
        image_datastore.load_local_corrected_image(tile=1, bit=0, return_future=False),
        committed[0],
    )
    open_group(path).validate(strict=True)


@pytest.mark.unit
def test_sofima_field_uses_xyz_pixels_and_patch_centered_lattice(image_datastore):
    image_datastore.voxel_size_zyx_um = (2, 1, 0.5)
    image_datastore.save_local_corrected_image(
        np.zeros((8, 10, 12), dtype=np.uint16), tile=0, round=0
    )
    path = image_datastore.tile_scene_path
    group = open_group(path)
    original = next(
        transform
        for transform in group.ome_metadata().scene.coordinateTransformations
        if transform.name == "tile0000/round002/deformable"
    )
    flow = np.empty((3, 2, 3, 3), dtype=np.float32)
    flow[0] = [1, 3, 5]
    flow[1] = -1
    flow[2] = 0.5

    image_datastore.save_local_sofima_flow_field(
        flow,
        tile=0,
        round=1,
        reference_round=0,
        map_stride_zyx_px=(2, 2, 4),
        map_box_start_xyz_px=(2, 1, 1),
        map_box_size_xyz_px=(8, 4, 2),
        reference_shape_zyx_px=(8, 10, 12),
        moving_shape_zyx_px=(8, 10, 12),
    )

    reopened = open_group(path).validate(strict=True)
    deformation = next(
        transform
        for transform in reopened.ome_metadata().scene.coordinateTransformations
        if transform.name == original.name
    )
    assert isinstance(deformation, v06.DisplacementsTransformation)
    systems = {
        system.name: system
        for system in reopened.ome_metadata().scene.coordinateSystems
    }
    assert [axis.name for axis in systems[deformation.output.name].axes] == list("xyz")
    assert deformation.path.startswith("coordinateTransformations/")
    field = reopened[deformation.path]
    multiscale = field.ome_metadata(strict=True).multiscales[0]
    assert [axis.type for axis in multiscale.axes] == [
        "displacement",
        "space",
        "space",
        "space",
    ]
    assert multiscale.axes[0].discrete is True
    assert [axis.name for axis in multiscale.axes] == ["d", "z", "y", "x"]
    assert multiscale.coordinateTransformations[0].mapAxis == [0, 3, 2, 1]
    dataset = multiscale.datasets[0]
    scale, translation = dataset.coordinateTransformations[0].transformations
    np.testing.assert_array_equal(scale.scale, [1, 2, 2, 4])
    np.testing.assert_array_equal(translation.translation, [0, 1, 1, 2])
    actual = field[dataset.path].to_tensorstore().read().result()
    np.testing.assert_array_equal(actual, flow)
    transforms = {
        t.name: t for t in reopened.ome_metadata().scene.coordinateTransformations
    }
    to_pixels = transforms[f"{original.name}/to_pixels"].transformations
    to_physical = transforms[f"{original.name}/to_physical"].transformations
    point_zyx_um = np.array([2, 1, 1], dtype=float)
    point_xyz_px = point_zyx_um[to_pixels[0].mapAxis] * to_pixels[1].scale
    np.testing.assert_array_equal(point_xyz_px, [2, 1, 1])
    displaced_xyz_px = point_xyz_px + actual[:, 0, 0, 0]
    displaced_xyz_um = displaced_xyz_px * to_physical[0].scale
    np.testing.assert_array_equal(displaced_xyz_um[to_physical[1].mapAxis], [3, 0, 1.5])
    assert reopened.attrs["qi2lab"]["transform_status"][original.name] == "calculated"
    restored, metadata = image_datastore.load_local_sofima_flow_field(
        tile=0, round=1, return_future=False
    )
    np.testing.assert_array_equal(restored, flow)
    np.testing.assert_array_equal(metadata["map_stride_zyx_px"], [2, 2, 4])
    np.testing.assert_array_equal(metadata["map_box_start_xyz_px"], [2, 1, 1])
    assert metadata["reference_shape_zyx_px"] == [8, 10, 12]

    # Reprocessing replaces the field and its lattice together while a reader
    # that opened the previous field continues to see its original data.
    previous, _ = image_datastore.load_local_sofima_flow_field(
        tile=0, round=1, return_future=None
    )
    replacement = -flow
    image_datastore.save_local_sofima_flow_field(
        replacement,
        tile=0,
        round=1,
        reference_round=0,
        map_stride_zyx_px=(1, 3, 2),
        map_box_start_xyz_px=(4, 3, 2),
        map_box_size_xyz_px=(4, 6, 1),
        reference_shape_zyx_px=(8, 10, 12),
        moving_shape_zyx_px=(8, 10, 12),
    )
    from merfish3danalysis.qi2labDataStore import qi2labDataStore

    datastore = qi2labDataStore(image_datastore.datastore_path, validate=False)
    updated, metadata = datastore.load_local_sofima_flow_field(
        tile=0, round=1, return_future=False
    )
    np.testing.assert_array_equal(updated, replacement)
    np.testing.assert_array_equal(metadata["map_stride_zyx_px"], [1, 3, 2])
    np.testing.assert_array_equal(metadata["map_box_start_xyz_px"], [4, 3, 2])
    np.testing.assert_array_equal(previous.read().result(), flow)

    scene = open_group(path).validate(strict=True).ome_metadata(strict=True)
    updated_deformation = next(
        transform
        for transform in scene.scene.coordinateTransformations
        if transform.name == original.name
    )
    assert updated_deformation.path != deformation.path
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    assert transforms[f"{original.name}/to_pixels"].output == updated_deformation.input
    assert (
        updated_deformation.output == transforms[f"{original.name}/to_physical"].input
    )
