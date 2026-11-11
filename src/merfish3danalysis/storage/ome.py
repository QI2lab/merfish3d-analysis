"""Create OME-NGFF Scenes through yaozarrs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from yaozarrs import ZarrGroup, open_group, v06
from yaozarrs.write.v06 import write_image, write_scene

from merfish3danalysis.storage._scene_update import update_scene

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class ConcurrentSceneUpdateError(RuntimeError):
    """A transform changed after a worker read it."""


def _check_revision(
    products: Mapping[str, Any], selection: str, expected: str | None
) -> None:
    """Prevent a worker from replacing an image published by another worker."""
    previous = products.get(selection)
    if (None if previous is None else previous["revision"]) != expected:
        raise ConcurrentSceneUpdateError(f"Image {selection!r} has changed.")


def _image_frame(
    scene: v06.Scene, image: v06.Image, frame: str
) -> v06.CoordinateSystem:
    """Verify that an image can be linked to its requested Scene frame."""
    systems = {system.name: system for system in scene.scene.coordinateSystems or []}
    if frame not in systems:
        raise ValueError(f"Unknown image frame {frame!r}.")
    intrinsic = image.multiscales[0].intrinsic_coordinate_system
    axes = [(axis.name, axis.type, axis.unit) for axis in intrinsic.axes]
    target_axes = [(axis.name, axis.type, axis.unit) for axis in systems[frame].axes]
    spatial = [axis for axis in axes if axis[0] not in ("t", "c")]
    if spatial != target_axes and spatial != target_axes[1:]:
        raise ValueError("Image and target coordinate systems must have matching axes.")
    return intrinsic


def _expected_transform_index(
    scene: v06.Scene,
    statuses: Mapping[str, str],
    versions: Mapping[str, int],
    expected: v06.CoordinateTransformation,
    expected_status: str,
    expected_revision: int,
) -> int:
    """Detect concurrent or intermediate updates before replacing a transform."""
    matches = [
        i
        for i, transform in enumerate(scene.scene.coordinateTransformations)
        if transform.name == expected.name
    ]
    if (
        len(matches) != 1
        or scene.scene.coordinateTransformations[matches[0]] != expected
        or statuses.get(expected.name) != expected_status
        or versions.get(expected.name) != expected_revision
    ):
        raise ConcurrentSceneUpdateError(f"Transform {expected.name!r} has changed.")
    return matches[0]


def build_acquisition_scene(
    tile_ids: Sequence[str], bits_by_round: Mapping[str, Sequence[str]]
) -> tuple[v06.Scene, dict[str, str]]:
    """Describe pending acquisition relationships with NGFF identity models.

    The first round anchors each tile to world through stage, camera, and global
    transforms. Other rounds retain their measured stage/camera relationships as
    separate acquisition frames, avoiding a second registration path to world.
    Chromatic edges point from a readout channel toward its round reference.
    The returned statuses are application state, not transform values.

    Parameters
    ----------
    tile_ids : Sequence[str]
        Stored tile identifiers.
    bits_by_round : Mapping[str, Sequence[str]]
        Ordered mapping from round identifiers to their readout bit identifiers.

    Returns
    -------
    tuple of (yaozarrs.v06.Scene, dict[str, str])
        Acquisition Scene and pending status for each transform.
    """
    tiles = list(tile_ids)
    rounds = list(bits_by_round)
    identifiers = [
        *tiles,
        *rounds,
        *(bit for bits in bits_by_round.values() for bit in bits),
    ]
    if not tiles or not rounds:
        raise ValueError("At least one tile and round are required.")
    if len(set(tiles)) != len(tiles):
        raise ValueError("Tile identifiers must be unique.")
    if any(
        not isinstance(value, str) or not value or "/" in value for value in identifiers
    ):
        raise ValueError("Identifiers must be nonempty strings without '/'.")
    if any(len(set(bits)) != len(bits) for bits in bits_by_round.values()):
        raise ValueError("Readout identifiers must be unique within a round.")

    axes = [v06.SpaceAxis(name=axis, unit="micrometer") for axis in "zyx"]
    systems = [v06.CoordinateSystem(name="world", axes=axes)]
    transforms: list[v06.IdentityTransformation] = []
    statuses: dict[str, str] = {}

    def add_system(name: str) -> str:
        """Add a shared frame for the acquisition transform graph."""
        systems.append(v06.CoordinateSystem(name=name, axes=axes))
        return name

    def add_identity(name: str, source: str, target: str) -> None:
        """Reserve a relationship until its registration is calculated."""
        transforms.append(
            v06.IdentityTransformation(
                name=name,
                input=v06.CoordinateSystemRef(name=source),
                output=v06.CoordinateSystemRef(name=target),
            )
        )
        statuses[name] = "pending"

    reference_round = rounds[0]
    for tile in tiles:
        reference = add_system(f"{tile}/reference")
        stage = add_system(f"{tile}/{reference_round}/after_stage")
        camera = add_system(f"{tile}/{reference_round}/after_camera")
        add_identity(f"{tile}/{reference_round}/stage", reference, stage)
        add_identity(f"{tile}/{reference_round}/camera", stage, camera)
        add_identity(f"{tile}/global", camera, "world")
        pixel_grid = f"{tile}/decoder_pixels"
        systems.append(
            v06.CoordinateSystem(
                name=pixel_grid, axes=[v06.SpaceAxis(name=axis) for axis in "zyx"]
            )
        )
        scaled_grid = add_system(f"{tile}/decoder_scaled")
        add_identity(f"{tile}/decoder_spacing", pixel_grid, scaled_grid)
        add_identity(f"{tile}/decoder_origin", scaled_grid, stage)

        for round_id, bits in bits_by_round.items():
            prefix = f"{tile}/{round_id}"
            deformed = add_system(f"{prefix}/deformed")
            moving = add_system(f"{prefix}/moving")
            add_identity(f"{prefix}/deformable", reference, deformed)
            add_identity(f"{prefix}/local", moving, deformed)
            if round_id != reference_round:
                measured_stage = add_system(f"{prefix}/after_stage")
                measured_camera = add_system(f"{prefix}/after_camera")
                add_identity(f"{prefix}/stage", moving, measured_stage)
                add_identity(f"{prefix}/camera", measured_stage, measured_camera)
            for bit in bits:
                native = add_system(f"{prefix}/{bit}/native")
                add_identity(f"{prefix}/{bit}/chromatic", native, moving)

    scene = v06.Scene(
        scene=v06.SceneDef(
            coordinateSystems=systems,
            coordinateTransformations=transforms,
        )
    )
    return scene, statuses


def initialize_scene(path: str | Path) -> Path:
    """Create an empty Scene before acquisition coordinates are known.

    The initial self-edge is a neutral identity, not a measured stage position.
    The writer refuses existing destinations so it cannot replace a live Scene.

    Parameters
    ----------
    path : str or Path
        Destination for the new Scene.

    Returns
    -------
    Path
        Path to the created Scene.
    """
    system = v06.CoordinateSystem(
        name="world",
        axes=[v06.SpaceAxis(name=axis, unit="micrometer") for axis in "zyx"],
    )
    reference = v06.CoordinateSystemRef(name=system.name)
    scene = v06.Scene(
        scene=v06.SceneDef(
            coordinateSystems=[system],
            coordinateTransformations=[
                v06.IdentityTransformation(input=reference, output=reference)
            ],
        )
    )
    return write_scene(
        path,
        scene,
        {},
        extra_attributes={
            "qi2lab": {
                "schema_version": "0.7",
                "dataset_id": uuid4().hex,
                "generation": 0,
                "products": {},
                "transform_status": {},
                "transform_versions": {},
                "transform_provenance": {},
            }
        },
        writer="tensorstore",
        overwrite=False,
    )


def publish_acquisition_scene(
    path: str | Path,
    tile_ids: Sequence[str],
    bits_by_round: Mapping[str, Sequence[str]],
    bit_linkers_by_round: Mapping[str, Sequence[int]],
    experiment_order: Sequence[Sequence[int]],
    channels_in_data: Sequence[Any],
) -> None:
    """Replace a newly initialized Scene with its pending acquisition graph.

    Parameters
    ----------
    path : str or Path
        Existing initialized Scene path.
    tile_ids : Sequence[str]
        Stored tile identifiers.
    bits_by_round : Mapping[str, Sequence[str]]
        Ordered round-to-readout identifier mapping.
    bit_linkers_by_round : Mapping[str, Sequence[int]]
        Per-round bit linkers to store for each tile.
    experiment_order : Sequence[Sequence[int]]
        Acquisition order, including empty channel slots.
    channels_in_data : Sequence[Any]
        Acquisition channel identifiers in order.
    """
    scene, statuses = build_acquisition_scene(tile_ids, bits_by_round)

    def update(
        current: v06.Scene, attributes: dict[str, Any]
    ) -> tuple[v06.Scene, dict[str, Any]]:
        """Publish the graph without discarding concurrently created products."""
        state = dict(attributes["qi2lab"])
        if state.get("tile_ids") or state.get("transform_status"):
            raise ConcurrentSceneUpdateError(
                "The acquisition Scene is already initialized."
            )
        products = state.get("products", {})
        if any(product.get("frame") != "world" for product in products.values()):
            raise ConcurrentSceneUpdateError(
                "Existing products must be in world coordinates."
            )
        product_edges = {product["edge"] for product in products.values()}
        retained = [
            t
            for t in current.scene.coordinateTransformations
            if t.name in product_edges
        ]
        candidate = scene.model_copy(
            update={
                "scene": scene.scene.model_copy(
                    update={
                        "coordinateTransformations": [
                            *scene.scene.coordinateTransformations,
                            *retained,
                        ]
                    }
                )
            }
        )
        entity_metadata = {
            f"fiducial/{tile}/{round_id}": {"bit_linker": list(values)}
            for tile in tile_ids
            for round_id, values in bit_linkers_by_round.items()
        }
        round_links: dict[str, str] = {}
        for tile in tile_ids:
            for round_id, bits in bits_by_round.items():
                for bit in bits:
                    round_links.setdefault(f"{tile}/{bit}", round_id)
        state.update(
            generation=int(state["generation"]) + 1,
            transform_status=statuses,
            transform_versions=dict.fromkeys(statuses, 0),
            reference_round=next(iter(bits_by_round)),
            tile_ids=list(tile_ids),
            bits_by_round={key: list(value) for key, value in bits_by_round.items()},
            experiment_order=[list(row) for row in experiment_order],
            channels_in_data=list(channels_in_data),
            entity_metadata=entity_metadata,
            round_links=round_links,
        )
        attributes["qi2lab"] = state
        return candidate, attributes

    update_scene(path, update)


def publish_transform(
    path: str | Path,
    expected: v06.CoordinateTransformation,
    replacement: v06.CoordinateTransformation,
    *,
    expected_status: str = "pending",
    expected_revision: int = 0,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Publish one completed transform without overwriting concurrent updates.

    Parameters
    ----------
    path : str or Path
        Acquisition Scene path.
    expected : yaozarrs.v06.CoordinateTransformation
        Previously read transform, used to detect concurrent updates.
    replacement : yaozarrs.v06.CoordinateTransformation
        New transform with the same name and coordinate frames.
    expected_status : str
        Previously read transform status.
    expected_revision : int
        Previously read transform revision.
    provenance : Mapping[str, Any] or None
        Optional provenance stored with the replacement.
    """
    publish_transforms(
        path,
        [expected],
        [replacement],
        expected_statuses={expected.name: expected_status},
        expected_revisions={expected.name: expected_revision},
        provenance=provenance,
    )


def publish_transforms(
    path: str | Path,
    expected: Sequence[v06.CoordinateTransformation],
    replacements: Sequence[v06.CoordinateTransformation],
    *,
    expected_statuses: Mapping[str, str],
    expected_revisions: Mapping[str, int],
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Publish related transforms atomically because yaozarrs cannot update a Scene.

    Parameters
    ----------
    path : str or Path
        Acquisition Scene path.
    expected : Sequence[yaozarrs.v06.CoordinateTransformation]
        Previously read transforms, used to detect concurrent updates.
    replacements : Sequence[yaozarrs.v06.CoordinateTransformation]
        New transforms in the same order, preserving names and frames.
    expected_statuses : Mapping[str, str]
        Previously read status for each transform name.
    expected_revisions : Mapping[str, int]
        Previously read revision for each transform name.
    provenance : Mapping[str, Any] or None
        Optional provenance stored with each replacement.
    """
    names = [transform.name for transform in expected]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Expected uniquely named transforms.")
    if (
        len(expected) != len(replacements)
        or set(expected_statuses) != set(names)
        or set(expected_revisions) != set(names)
    ):
        raise ValueError("Every transform needs a replacement, status, and revision.")
    if any(
        replacement.name != original.name
        or replacement.input != original.input
        or replacement.output != original.output
        for original, replacement in zip(expected, replacements, strict=True)
    ):
        raise ValueError("A replacement must preserve the transform name and frames.")
    if any(name.endswith("/chromatic") for name in names):
        raise ValueError(
            "Use an optimizer batch to update all applicable chromatic relationships."
        )

    def update(
        scene: v06.Scene, attributes: dict[str, Any]
    ) -> tuple[v06.Scene, dict[str, Any]]:
        """Replace transforms only when their expected revisions still match."""
        state = dict(attributes["qi2lab"])
        statuses = dict(state["transform_status"])
        versions = dict(state["transform_versions"])
        provenance_by_transform = dict(state.get("transform_provenance", {}))
        transforms = list(scene.scene.coordinateTransformations)
        indices = [
            _expected_transform_index(
                scene,
                statuses,
                versions,
                original,
                expected_statuses[original.name],
                expected_revisions[original.name],
            )
            for original in expected
        ]
        for index, original, replacement in zip(
            indices, expected, replacements, strict=True
        ):
            transforms[index] = replacement
            statuses[original.name] = "calculated"
            versions[original.name] = expected_revisions[original.name] + 1
            if provenance is None:
                provenance_by_transform.pop(original.name, None)
            else:
                provenance_by_transform[original.name] = dict(provenance)
        candidate = scene.model_copy(
            update={
                "scene": scene.scene.model_copy(
                    update={"coordinateTransformations": transforms}
                )
            }
        )
        state.update(
            generation=int(state["generation"]) + 1,
            transform_status=statuses,
            transform_versions=versions,
            transform_provenance=provenance_by_transform,
        )
        attributes["qi2lab"] = state
        return candidate, attributes

    update_scene(path, update)


def update_chromatic_metadata(
    path: str | Path,
    *,
    calibration: Mapping[str, Any] | None = None,
    wavelength: tuple[str, str, tuple[float, float]] | None = None,
) -> None:
    """Update wavelength links atomically; yaozarrs cannot amend an existing Scene.

    Parameters
    ----------
    path : str or Path
        Acquisition Scene path.
    calibration : Mapping[str, Any] or None
        Updated optimizer chromatic calibration, if supplied.
    wavelength : tuple or None
        Tile identifier, bit identifier, and excitation/emission wavelengths
        in micrometers, if updating a readout link.
    """
    from merfish3danalysis.utils.decode_warping import (
        optimizer_calibration_is_applicable,
        select_chromatic_affine,
        validate_chromatic_affine,
    )

    def update(
        scene: v06.Scene, attributes: dict[str, Any]
    ) -> tuple[v06.Scene, dict[str, Any]]:
        """Keep all channel relationships consistent with optimizer metadata."""
        state = dict(attributes["qi2lab"])
        wavelengths = dict(state.get("readout_wavelengths_um", {}))
        if wavelength is not None:
            tile, bit, values = wavelength
            wavelengths[f"{tile}/{bit}"] = list(values)
        if calibration is not None:
            state["chromatic_calibration"] = dict(calibration)
        current_calibration = state.get("chromatic_calibration", {})
        applicable = optimizer_calibration_is_applicable(current_calibration)
        statuses = dict(state.get("transform_status", {}))
        versions = dict(state.get("transform_versions", {}))
        provenance = dict(state.get("transform_provenance", {}))
        transforms = []
        for transform in scene.scene.coordinateTransformations:
            if not transform.name or not transform.name.endswith("/chromatic"):
                transforms.append(transform)
                continue
            tile, _, bit, _ = transform.name.split("/")
            if (
                wavelength is not None
                and calibration is None
                and (tile, bit) != wavelength[:2]
            ):
                transforms.append(transform)
                continue
            pair = wavelengths.get(f"{tile}/{bit}")
            matrix = None
            if pair is not None and applicable:
                try:
                    matrix = validate_chromatic_affine(
                        select_chromatic_affine(
                            current_calibration, wavelength_um=pair[1]
                        )
                    )
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    pass
            if matrix is None:
                replacement = v06.IdentityTransformation(
                    name=transform.name, input=transform.input, output=transform.output
                )
                statuses[transform.name] = "pending"
                provenance.pop(transform.name, None)
            else:
                replacement = v06.AffineTransformation(
                    name=transform.name,
                    input=transform.input,
                    output=transform.output,
                    affine=matrix[:3].tolist(),
                )
                statuses[transform.name] = "calculated"
                provenance[transform.name] = {
                    "estimator": current_calibration["estimator"],
                    "wavelength_um": pair[1],
                }
            if replacement != transform:
                versions[transform.name] = int(versions.get(transform.name, 0)) + 1
            transforms.append(replacement)
        state.update(
            generation=int(state["generation"]) + 1,
            readout_wavelengths_um=wavelengths,
            transform_status=statuses,
            transform_versions=versions,
            transform_provenance=provenance,
        )
        attributes["qi2lab"] = state
        return scene.model_copy(
            update={
                "scene": scene.scene.model_copy(
                    update={"coordinateTransformations": transforms}
                )
            }
        ), attributes

    update_scene(path, update)


def publish_sofima_field(
    path: str | Path,
    expected: v06.CoordinateTransformation,
    field_xyz_px: Any,
    *,
    spacing_zyx_um: Sequence[float],
    stride_zyx_px: Sequence[float],
    start_xyz_px: Sequence[float],
    expected_status: str = "pending",
    expected_revision: int = 0,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Store the existing XYZ-pixel field using NGFF displacement models.

    Parameters
    ----------
    path : str or Path
        Acquisition Scene path.
    expected : yaozarrs.v06.CoordinateTransformation
        Previously read deformable transform.
    field_xyz_px : Any
        Displacement field with XYZ pixel components and ZYX spatial axes.
    spacing_zyx_um : Sequence[float]
        Image pixel spacing in ZYX micrometers.
    stride_zyx_px : Sequence[float]
        Field lattice stride in ZYX pixels.
    start_xyz_px : Sequence[float]
        Field lattice origin in XYZ pixels.
    expected_status : str
        Previously read transform status.
    expected_revision : int
        Previously read transform revision.
    extra_attributes : Mapping[str, Any] or None
        Additional metadata stored with the field image.

    """
    if not expected.name or not expected.name.endswith("/deformable"):
        raise ValueError("A SOFIMA field must replace a deformable relationship.")
    field = np.asarray(field_xyz_px, dtype=np.float32)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    stride = np.asarray(stride_zyx_px, dtype=np.float64)
    start = np.asarray(start_xyz_px, dtype=np.float64)
    if field.ndim != 4 or field.shape[0] != 3 or any(n == 0 for n in field.shape):
        raise ValueError("A SOFIMA field must have shape (3, z, y, x).")
    if any(
        values.shape != (3,) or not np.isfinite(values).all()
        for values in (spacing, stride, start)
    ):
        raise ValueError(
            "Field spacing, stride and start must have three finite values."
        )
    if np.any(spacing <= 0) or np.any(stride <= 0):
        raise ValueError("Field spacing and stride must be positive.")

    group = open_group(path)
    current = group.ome_metadata(strict=True)
    if not isinstance(current, v06.Scene):
        raise ValueError("Expected an NGFF 0.6 Scene.")
    _expected_transform_index(
        current,
        group.attrs["qi2lab"]["transform_status"],
        group.attrs["qi2lab"]["transform_versions"],
        expected,
        expected_status,
        expected_revision,
    )
    to_pixels_name = f"{expected.name}/to_pixels"
    to_physical_name = f"{expected.name}/to_physical"
    input_pixels = f"{expected.name}/input_xyz_px"
    output_pixels = f"{expected.name}/output_xyz_px"
    if isinstance(expected, v06.IdentityTransformation):
        physical_input, physical_output = expected.input, expected.output
    elif isinstance(expected, v06.DisplacementsTransformation):
        transforms = {t.name: t for t in current.scene.coordinateTransformations}
        physical_input = transforms[to_pixels_name].input
        physical_output = transforms[to_physical_name].output
    else:
        raise ValueError("SOFIMA must replace an identity or displacement transform.")

    system = v06.CoordinateSystem(
        name="intrinsic",
        axes=[
            v06.CustomAxis(name="d", type="displacement", discrete=True),
            *[v06.SpaceAxis(name=axis) for axis in "zyx"],
        ],
    )
    xyz_system = v06.CoordinateSystem(
        name="xyz_lattice",
        axes=[
            v06.CustomAxis(name="d", type="displacement", discrete=True),
            *[v06.SpaceAxis(name=axis) for axis in "xyz"],
        ],
    )
    dataset_path = "field"
    image = v06.Image(
        multiscales=[
            v06.Multiscale(
                coordinateSystems=[system, xyz_system],
                coordinateTransformations=[
                    v06.MapAxisTransformation(
                        input=v06.CoordinateSystemRef(name=system.name),
                        output=v06.CoordinateSystemRef(name=xyz_system.name),
                        mapAxis=[0, 3, 2, 1],
                    )
                ],
                datasets=[
                    v06.Dataset(
                        path=dataset_path,
                        coordinateTransformations=[
                            v06.SequenceTransformation(
                                input=v06.CoordinateSystemRef(path=dataset_path),
                                output=v06.CoordinateSystemRef(name=system.name),
                                transformations=[
                                    v06.ScaleTransformation(
                                        scale=[1, *stride.tolist()]
                                    ),
                                    v06.TranslationTransformation(
                                        translation=[0, *start[::-1].tolist()]
                                    ),
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    from yaozarrs.write._common._groups import _ensure_parent_groups

    relative_path = f"coordinateTransformations/field_{uuid4().hex}"
    _ensure_parent_groups(Path(path), relative_path)
    options = {
        "writer": "tensorstore",
        "extra_attributes": dict(extra_attributes) if extra_attributes else None,
    }
    write_image(Path(path) / relative_path, image, field, **options)
    spacing_xyz = spacing[::-1]
    to_pixels = v06.SequenceTransformation(
        name=to_pixels_name,
        input=physical_input,
        output=v06.CoordinateSystemRef(name=input_pixels),
        transformations=[
            v06.MapAxisTransformation(mapAxis=[2, 1, 0]),
            v06.ScaleTransformation(scale=(1 / spacing_xyz).tolist()),
        ],
    )
    deformation = v06.DisplacementsTransformation(
        name=expected.name,
        input=v06.CoordinateSystemRef(name=input_pixels),
        output=v06.CoordinateSystemRef(name=output_pixels),
        path=relative_path,
    )
    to_physical = v06.SequenceTransformation(
        name=to_physical_name,
        input=v06.CoordinateSystemRef(name=output_pixels),
        output=physical_output,
        transformations=[
            v06.ScaleTransformation(scale=spacing_xyz.tolist()),
            v06.MapAxisTransformation(mapAxis=[2, 1, 0]),
        ],
    )

    def update(
        scene: v06.Scene, attributes: dict[str, Any]
    ) -> tuple[v06.Scene, dict[str, Any]]:
        """Publish the field and its pixel coordinate systems atomically."""
        state = dict(attributes["qi2lab"])
        statuses = dict(state["transform_status"])
        versions = dict(state["transform_versions"])
        _expected_transform_index(
            scene, statuses, versions, expected, expected_status, expected_revision
        )
        systems = list(scene.scene.coordinateSystems)
        existing_names = {system.name for system in systems}
        for name in (input_pixels, output_pixels):
            if name not in existing_names:
                systems.append(
                    v06.CoordinateSystem(
                        name=name, axes=[v06.SpaceAxis(name=axis) for axis in "xyz"]
                    )
                )
        transforms = [
            t
            for t in scene.scene.coordinateTransformations
            if t.name not in {expected.name, to_pixels_name, to_physical_name}
        ]
        transforms.extend((to_pixels, deformation, to_physical))
        statuses[expected.name] = "calculated"
        versions[expected.name] = expected_revision + 1
        state.update(
            generation=int(state["generation"]) + 1,
            transform_status=statuses,
            transform_versions=versions,
        )
        attributes["qi2lab"] = state
        return scene.model_copy(
            update={
                "scene": scene.scene.model_copy(
                    update={
                        "coordinateSystems": systems,
                        "coordinateTransformations": transforms,
                    }
                )
            }
        ), attributes

    update_scene(path, update)


def publish_image(
    path: str | Path,
    selection: str,
    frame: str,
    image: v06.Image,
    pixels: Any,
    *,
    expected_revision: str | None = None,
    chunks: Sequence[int] | None = None,
    shards: Sequence[int] | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
    affine_zyx_um: Any | None = None,
    missing_z_origin_um: float = 0,
) -> str:
    """Publish an image only after its TensorStore write has completed.

    Parameters
    ----------
    path : str or Path
        Destination Scene path.
    selection : str
        Logical product identifier in the Scene.
    frame : str
        Scene coordinate system to which the image is linked.
    image : yaozarrs.v06.Image
        Image metadata model.
    pixels : Any
        Image pixel data passed to the TensorStore writer.
    expected_revision : str or None
        Previously published product revision, used to detect concurrent writes.
    chunks : Sequence[int] or None
        Inner chunk shape; None selects the writer default.
    shards : Sequence[int] or None
        Shard shape, or None for unsharded storage.
    extra_attributes : Mapping[str, Any] or None
        Additional image metadata.
    affine_zyx_um : Any or None
        Optional 3D affine from the image coordinates to its Scene frame.
    missing_z_origin_um : float
        Z coordinate used when linking a 2D image into a 3D frame.

    Returns
    -------
    str
        Published revision.
    """
    if not selection:
        raise ValueError("An image selection is required.")
    current = open_group(path)
    current_scene = current.ome_metadata(strict=True)
    if not isinstance(current_scene, v06.Scene):
        raise ValueError("Expected an NGFF 0.6 Scene.")
    _check_revision(current.attrs["qi2lab"]["products"], selection, expected_revision)
    _image_frame(current_scene, image, frame)
    revision = uuid4().hex
    relative_path = f"image_{revision}.ome.zarr"
    options = {
        "writer": "tensorstore",
        "chunks": tuple(chunks) if chunks is not None else "auto",
        "shards": tuple(shards) if shards is not None else None,
        "extra_attributes": dict(extra_attributes) if extra_attributes else None,
    }
    write_image(Path(path) / relative_path, image, pixels, **options)

    publish_completed_image(
        path,
        selection,
        frame,
        image,
        revision,
        expected_revision=expected_revision,
        affine_zyx_um=affine_zyx_um,
        missing_z_origin_um=missing_z_origin_um,
    )
    return revision


def publish_completed_image(
    path: str | Path,
    selection: str,
    frame: str,
    image: v06.Image,
    revision: str,
    *,
    expected_revision: str | None = None,
    affine_zyx_um: Any | None = None,
    missing_z_origin_um: float = 0,
) -> None:
    """Publish completed pixels; yaozarrs cannot update an existing Scene.

    Parameters
    ----------
    path : str or Path
        Destination Scene path.
    selection : str
        Logical product identifier in the Scene.
    frame : str
        Scene coordinate system to which the image is linked.
    image : yaozarrs.v06.Image
        Metadata for the completed image.
    revision : str
        Revision used in the completed image path.
    expected_revision : str or None
        Previously published product revision, used to detect concurrent writes.
    affine_zyx_um : Any or None
        Optional 3D affine from the image coordinates to its Scene frame.
    missing_z_origin_um : float
        Z coordinate used when linking a 2D image into a 3D frame.
    """
    relative_path = f"image_{revision}.ome.zarr"
    edge_name = f"image/{revision}"

    def update(
        scene: v06.Scene, attributes: dict[str, Any]
    ) -> tuple[v06.Scene, dict[str, Any]]:
        """Publish a completed image and its frame together under the Scene lock."""
        state = dict(attributes["qi2lab"])
        products = dict(state["products"])
        previous = products.get(selection)
        _check_revision(products, selection, expected_revision)
        intrinsic = _image_frame(scene, image, frame)
        transforms = list(scene.scene.coordinateTransformations)
        if previous is not None:
            transforms = [t for t in transforms if t.name != previous["edge"]]
        source = v06.CoordinateSystemRef(name=intrinsic.name, path=relative_path)
        target = v06.CoordinateSystemRef(name=frame)
        axes = [axis.name for axis in intrinsic.axes]
        dropped = [index for index, axis in enumerate(axes) if axis in ("t", "c")]
        target_axes = next(
            system.axes
            for system in scene.scene.coordinateSystems
            if system.name == frame
        )
        created = [0] if "z" not in axes and target_axes[0].name == "z" else None
        steps: list[v06.CoordinateTransformation] = []
        if dropped or created:
            steps.append(
                v06.ProjectAxisTransformation(
                    droppedInputs=dropped or None, createdOutputs=created
                )
            )
        if created and missing_z_origin_um:
            steps.append(
                v06.TranslationTransformation(
                    translation=[float(missing_z_origin_um), 0, 0]
                )
            )
        if affine_zyx_um is not None:
            affine = np.asarray(affine_zyx_um, dtype=np.float32)
            if (
                affine.shape != (4, 4)
                or not np.isfinite(affine).all()
                or not np.array_equal(affine[3], [0, 0, 0, 1])
            ):
                raise ValueError(
                    "Image affine must be a finite 4x4 homogeneous matrix."
                )
            steps.append(v06.AffineTransformation(affine=affine[:3].tolist()))
        if not steps:
            link = v06.IdentityTransformation(
                name=edge_name, input=source, output=target
            )
        elif len(steps) == 1:
            link = steps[0].model_copy(
                update={"name": edge_name, "input": source, "output": target}
            )
        else:
            link = v06.SequenceTransformation(
                name=edge_name, input=source, output=target, transformations=steps
            )
        transforms.append(link)
        candidate = scene.model_copy(
            update={
                "scene": scene.scene.model_copy(
                    update={"coordinateTransformations": transforms}
                )
            }
        )
        products[selection] = {
            "path": relative_path,
            "revision": revision,
            "edge": edge_name,
            "frame": frame,
        }
        state.update(generation=int(state["generation"]) + 1, products=products)
        attributes["qi2lab"] = state
        return candidate, attributes

    update_scene(path, update)


def publish_projection_labels(path: str | Path, labels: Any) -> None:
    """Stage labels before publication because yaozarrs lists them before writing.

    Parameters
    ----------
    path : str or Path
        Fiducial projection Scene path.
    labels : Any
        Integer YX labels on the projection grid.

    """
    import os
    import shutil
    from tempfile import TemporaryDirectory

    from yaozarrs.write._common._groups import _update_zarr3_group
    from yaozarrs.write.v06 import LabelsBuilder

    from merfish3danalysis.storage._scene_update import _scene_lock

    path = Path(path)
    root = open_group(path)
    product = root.attrs["qi2lab"]["products"]["projection/fiducial"]
    image_path = path / product["path"]
    target = image_path / "labels"
    previous_labels = (
        open_group(target).ome_metadata(strict=True).labels if target.exists() else []
    )
    source = open_group(image_path)
    model = source.ome_metadata(strict=True)
    dataset = model.multiscales[0].datasets[0]
    masks = np.asarray(labels)
    if not np.issubdtype(masks.dtype, np.integer) or tuple(masks.shape) != tuple(
        source[dataset.path].metadata.shape
    ):
        raise ValueError(
            "Cellpose labels must be an integer YX image on the projection grid."
        )
    label_model = v06.LabelImage(
        multiscales=model.multiscales,
        image_label=v06.ImageLabel(source=v06.LabelSource(image="../../")),
    )
    name = f"cellpose_{uuid4().hex}.ome.zarr"
    with TemporaryDirectory(prefix=f".{path.name}.labels-", dir=path.parent) as temp:
        staged_image = Path(temp) / "image"
        staged_image.mkdir()
        shutil.copy2(image_path / "zarr.json", staged_image / "zarr.json")
        for child in image_path.iterdir():
            if child.name not in {"zarr.json", "labels"}:
                (staged_image / child.name).symlink_to(child.resolve())
        staged_labels = staged_image / "labels"
        builder = LabelsBuilder(staged_labels, writer="tensorstore")
        builder.write_label(name, label_model, masks)
        open_group(staged_image).validate(strict=True)

        with _scene_lock(path.resolve()):
            current = open_group(path).attrs["qi2lab"]["products"][
                "projection/fiducial"
            ]
            if current["revision"] != product["revision"]:
                raise ConcurrentSceneUpdateError("The fiducial projection has changed.")
            current_labels = (
                open_group(target).ome_metadata(strict=True).labels
                if target.exists()
                else []
            )
            if current_labels != previous_labels:
                raise ConcurrentSceneUpdateError("The projection labels have changed.")
            if not target.exists():
                os.rename(staged_labels, target)
            else:
                existing = open_group(target)
                old_model = existing.ome_metadata(strict=True)
                names = [
                    item
                    for item in old_model.labels
                    if not item.startswith("cellpose_")
                ]
                names.append(name)
                os.rename(staged_labels / name, target / name)
                _update_zarr3_group(target, v06.LabelsGroup(labels=names))
                with (target / "zarr.json").open("rb") as stream:
                    os.fsync(stream.fileno())
                directory = os.open(target, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)


def _axis_aligned_grid(
    dataset: v06.Dataset, ndim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read the dataset grid through yaozarrs and require valid sampling spacing."""
    scale = dataset.scale_transform
    translation = dataset.translation_transform
    spacing = (
        np.ones(ndim) if scale is None else np.asarray(scale.scale, dtype=np.float64)
    )
    origin = (
        np.zeros(ndim)
        if translation is None
        else np.asarray(translation.translation, dtype=np.float64)
    )
    if (
        spacing.shape != (ndim,)
        or origin.shape != (ndim,)
        or not np.all(np.isfinite(spacing))
        or not np.all(np.isfinite(origin))
        or np.any(spacing <= 0)
    ):
        raise ValueError("Expected a positive axis-aligned image grid.")
    return spacing, origin


def load_sofima_field(
    path: str | Path | ZarrGroup,
    transform_name: str,
    *,
    return_future: bool | None = False,
) -> tuple[Any, dict[str, Any]] | None:
    """Read the NGFF field in the SOFIMA API's existing XYZ/ZYX convention.

    Parameters
    ----------
    path : str, Path, or yaozarrs.ZarrGroup
        Acquisition Scene path or open group.
    transform_name : str
        Name of the deformable Scene transform.
    return_future : bool or None
        True returns a TensorStore read future, False a NumPy array, and None
        a sliceable TensorStore handle.

    Returns
    -------
    tuple or None
        Field and metadata, or None if the transform is still identity.
    """
    group = open_group(path)
    scene = group.ome_metadata(strict=True)
    if not isinstance(scene, v06.Scene):
        raise ValueError("Expected an NGFF 0.6 Scene.")
    transforms = {t.name: t for t in scene.scene.coordinateTransformations}
    transform = transforms[transform_name]
    if isinstance(transform, v06.IdentityTransformation):
        return None
    if not isinstance(transform, v06.DisplacementsTransformation):
        raise ValueError(f"Transform {transform_name!r} is not a displacement field.")
    field = group[transform.path]
    field_model = field.ome_metadata(strict=True)
    if not isinstance(field_model, v06.Image):
        raise ValueError("The displacement field must be an NGFF image.")
    field_multiscale = field_model.multiscales[0]
    axes = field_multiscale.axes
    if [axis.type for axis in axes] != ["displacement", "space", "space", "space"] or [
        axis.name for axis in axes[1:]
    ] != list("zyx"):
        raise ValueError("Expected a displacement image with axes (d, z, y, x).")
    field_dataset = field_multiscale.datasets[0]
    field_spacing, field_origin = _axis_aligned_grid(field_dataset, 4)
    if not np.isclose(field_spacing[0], 1) or not np.isclose(field_origin[0], 0):
        raise ValueError("The displacement axis must have unit scale and zero origin.")
    field_array = field[field_dataset.path]
    dimension_names = getattr(field_array.metadata, "dimension_names", None)
    if dimension_names is not None and list(dimension_names) != [
        axis.name for axis in axes
    ]:
        raise ValueError(
            "Displacement array dimension names do not match the NGFF axes."
        )
    lazy = field_array.to_tensorstore()
    if lazy.ndim != 4 or lazy.shape[0] != 3:
        raise ValueError("Expected three XYZ displacement components.")
    metadata = dict(field.attrs)
    metadata.pop("ome", None)
    metadata.update(
        map_stride_zyx_px=field_spacing[1:].tolist(),
        map_box_start_xyz_px=field_origin[1:][::-1].tolist(),
    )
    if return_future is None:
        return lazy, metadata
    future = lazy.read()
    return (future if return_future else future.result()), metadata
