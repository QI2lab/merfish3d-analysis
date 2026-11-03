"""
Interface to qi2lab MERFISH datastore.

This module provides methods and attributes to create or interact with
the qi2lab MERFISH datastore. The filestore structure is further described
in the merfish3d-analysis documentation.

History:
---------
- **2024/12**: Refactored repo structure.
- **2024/12**: Updated docstrings and exception types.
- **2024/07**: Initial commit.
"""

import gzip
import json
import re
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import TimeoutError
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from yaozarrs import DimSpec, open_group, v06
from yaozarrs.write.v06 import write_image

from merfish3danalysis.utils.spacing import (
    round_pixel_size_um,
    round_spacing_um,
)

if TYPE_CHECKING:
    from tensorstore import Future, TensorStore, WriteFutures


class qi2labDataStore:
    """API to qi2lab MERFISH store.

    Parameters
    ----------
    datastore_path : str or Path
        Path to qi2lab MERFISH store
    validate : bool, default True
        Validate datastore contents on open. Set False to skip expensive
        zarr readability checks and load only metadata/path structure.

    """

    _FUSED_SCENE_SELECTIONS: ClassVar[dict[str, str]] = {
        "fiducial": "fused/fiducial",
        "all_channels": "fused/full_dataset",
    }

    def __init__(self, datastore_path: str | Path, validate: bool = True) -> None:
        """
        Initialize a qi2lab datastore handle.

        Parameters
        ----------
        datastore_path : str | Path
            Datastore root path.
        validate : bool
            Whether to validate existing image arrays while parsing the store.
        """
        self._datastore_path = Path(datastore_path)
        self._tile_scene_path = self._datastore_path / "tiles.ome.zarr"
        if self._datastore_path.exists():
            self._parse_datastore(validate=validate)
        else:
            self._init_datastore()

    @property
    def datastore_path(self) -> Path:
        """Experiment datastore directory."""
        return self._datastore_path

    @property
    def tile_scene_path(self) -> Path:
        """OME-NGFF Scene containing the native tile images and transforms."""
        return self._tile_scene_path

    def fiducial_projection_path(self) -> Path:
        """Return the Scene containing the fiducial projection and labels."""
        return self._datastore_path / "fiducial_projection.ome.zarr"

    def fused_scene_path(self, fusion_type: str = "fiducial") -> Path:
        """Return the separate Scene for a fiducial or full-data fusion.

        Parameters
        ----------
        fusion_type : str
            ``"fiducial"`` for the downsampled fiducial volume or
            ``"all_channels"`` for the full-data fusion.

        Returns
        -------
        Path
            Scene path; the Scene is not created by this method.
        """
        names = {
            "fiducial": "fiducial_fusion.ome.zarr",
            "all_channels": "fused.ome.zarr",
        }
        try:
            return self._datastore_path / names[fusion_type]
        except KeyError as exc:
            raise ValueError(
                "fusion_type must be 'fiducial' or 'all_channels'."
            ) from exc

    def _ensure_fused_scene(self, fusion_type: str) -> Path:
        """Atomically create a product Scene because MVS writes its images later."""
        import os
        from tempfile import TemporaryDirectory

        from merfish3danalysis.storage._scene_update import _scene_lock
        from merfish3danalysis.storage.ome import initialize_scene

        path = self.fused_scene_path(fusion_type)
        if not path.exists():
            with TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temp:
                candidate = Path(temp) / path.name
                initialize_scene(candidate)
                with _scene_lock(path.resolve()):
                    if not path.exists():
                        os.rename(candidate, path)
        return path

    def fused_image_path(self, image_name: str | None = None) -> Path:
        """Return the OME-Zarr path for a fused image.

        Parameters
        ----------
        image_name : str or None
            Logical image name. None selects the fused fiducial volume.

        Returns
        -------
        Path
            Fused image directory, without opening or creating it.
        """
        fusion_type = "fiducial" if image_name is None else "all_channels"
        scene_path = self.fused_scene_path(fusion_type)
        if scene_path.exists():
            products = open_group(scene_path).attrs["qi2lab"]["products"]
            product = products.get(self._FUSED_SCENE_SELECTIONS[fusion_type])
            if product is not None:
                return scene_path / product["path"]
        return scene_path / "image.ome.zarr"

    def prepare_global_fused_image(
        self, fusion_type: str
    ) -> tuple[Path, str, str | None]:
        """Stage an MVS fusion and capture the revision needed for safe publication.

        MVS writes blocks directly, so the Scene must reference the output only
        after all blocks are complete. Concurrent writers use unique image paths.

        Parameters
        ----------
        fusion_type : str
            ``"fiducial"`` or ``"all_channels"``.

        Returns
        -------
        tuple of (Path, str, str or None)
            New image path, new revision, and the previous published revision.
        """
        from uuid import uuid4

        try:
            selection = self._FUSED_SCENE_SELECTIONS[fusion_type]
        except KeyError as exc:
            raise ValueError(
                "fusion_type must be 'fiducial' or 'all_channels'."
            ) from exc
        scene_path = self._ensure_fused_scene(fusion_type)
        products = open_group(scene_path).attrs["qi2lab"]["products"]
        previous = products.get(selection)
        revision = uuid4().hex
        return (
            scene_path / f"image_{revision}.ome.zarr",
            revision,
            None if previous is None else previous["revision"],
        )

    def publish_global_fused_image(
        self,
        fusion_type: str,
        image: v06.Image,
        revision: str,
        expected_revision: str | None,
    ) -> None:
        """Publish completed MVS blocks atomically; yaozarrs cannot resume a Scene writer.

        Parameters
        ----------
        fusion_type : str
            ``"fiducial"`` or ``"all_channels"``.
        image : yaozarrs.v06.Image
            Metadata for the completed image.
        revision : str
            Revision returned when the image was staged.
        expected_revision : str or None
            Previously published revision, used to detect concurrent updates.
        """
        from merfish3danalysis.storage.ome import publish_completed_image

        try:
            selection = self._FUSED_SCENE_SELECTIONS[fusion_type]
        except KeyError as exc:
            raise ValueError(
                "fusion_type must be 'fiducial' or 'all_channels'."
            ) from exc
        publish_completed_image(
            self.fused_scene_path(fusion_type),
            selection,
            "world",
            image,
            revision,
            expected_revision=expected_revision,
        )

    def local_image_path(
        self,
        tile: int | str,
        image_name: str,
        *,
        round: int | str | None = None,
        bit: int | str | None = None,
    ) -> Path:
        """Return a native fiducial or readout OME-Zarr image path.

        Parameters
        ----------
        tile : int or str
            Zero-based tile index or stored tile identifier.
        image_name : str
            Logical image name such as corrected_data or decon_data.
        round : int or str or None
            Zero-based fiducial round index or identifier; excludes bit.
        bit : int or str or None
            Zero-based readout bit index or identifier; excludes round.

        Returns
        -------
        Path
            Image path without loading pixels or creating directories.
        """
        if (round is None) == (bit is None):
            raise ValueError("Provide exactly one of round or bit.")
        if isinstance(tile, int) and not 0 <= tile < len(self._tile_ids):
            raise ValueError(f"Invalid tile: {tile}")
        tile_id = self._tile_ids[tile] if isinstance(tile, int) else tile
        if tile_id not in self._tile_ids:
            raise ValueError(f"Invalid tile: {tile}")
        ids = self._round_ids if round is not None else self._bit_ids
        selection = round if round is not None else bit
        if isinstance(selection, int) and not 0 <= selection < len(ids):
            raise ValueError(f"Invalid round or bit: {selection}")
        entity_id = ids[selection] if isinstance(selection, int) else selection
        if entity_id not in ids:
            raise ValueError(f"Invalid round or bit: {selection}")
        state = open_group(self._tile_scene_path).attrs.get("qi2lab", {})
        product = state.get("products", {}).get(f"{tile_id}/{entity_id}/{image_name}")
        if product is not None:
            return self._tile_scene_path / product["path"]
        root = (
            self._fiducial_root_path if round is not None else self._readouts_root_path
        )
        return self._image_store_path(root / tile_id / entity_id / image_name)

    def load_local_fusion_image(
        self,
        tile: int | str,
        *,
        round: int | str | None = None,
        bit: int | str | None = None,
        chunks: Sequence[int] | None = None,
    ) -> Any:
        """Return the native Zarr handle required by MVS direct-Zarr fusion.

        Deconvolved pixels take precedence when present, matching global
        registration and fusion's existing input selection. Other datastore
        array access uses TensorStore. ``chunks`` requests a compatible lazy
        chunk view when mixing products with different storage layouts.

        Parameters
        ----------
        tile : int or str
            Zero-based tile index or stored tile identifier.
        round : int or str or None
            Fiducial round index or identifier; excludes bit.
        bit : int or str or None
            Readout bit index or identifier; excludes round.
        chunks : Sequence[int] or None
            Chunk shape for a lazy TensorStore-backed view, or None to use the
            native layout when supported.

        Returns
        -------
        zarr.Array
            Lazy array accepted by multiview-stitcher.
        """
        for image_name in ("decon_data", "corrected_data"):
            path = self.local_image_path(tile, image_name, round=round, bit=bit)
            if path.exists():
                from merfish3danalysis.storage.mvs import mvs_array

                group = open_group(path)
                image = group.ome_metadata(strict=True)
                return mvs_array(group[image.multiscales[0].datasets[0].path], chunks)
        raise FileNotFoundError(
            f"Missing fusion input for tile={tile} round={round} bit={bit}."
        )

    def local_feature_predictor_spots_path(self, tile: str, bit: str) -> Path:
        """Return the localization table path for a tile and readout identifier.

        Parameters
        ----------
        tile : str
            Stored tile identifier.
        bit : str
            Stored readout bit identifier.

        Returns
        -------
        Path
            Localization Parquet path; the file may not yet exist.
        """
        if tile not in self._tile_ids or bit not in self._bit_ids:
            raise ValueError(f"Invalid tile or bit: {tile}, {bit}")
        return (
            self._feature_predictor_localizations_root_path / tile / (bit + ".parquet")
        )

    def load_local_image_metadata(
        self,
        tile: int | str,
        *,
        round: int | str | None = None,
        bit: int | str | None = None,
        image_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Load Scene and image metadata for one fiducial round or readout bit.

        Parameters
        ----------
        tile : int or str
            Tile index or identifier.
        round : int or str or None
            Fiducial round index or identifier; excludes bit.
        bit : int or str or None
            Readout bit index or identifier; excludes round.
        image_names : Sequence[str] or None
            Image names to inspect, in metadata precedence order.

        Returns
        -------
        dict[str, Any]
            Merged image metadata, with Scene entity values taking precedence.
        """
        self.local_image_path(tile, "corrected_data", round=round, bit=bit)
        tile_id = self._tile_ids[tile] if isinstance(tile, int) else tile
        if round is not None:
            entity_id = self._round_ids[round] if isinstance(round, int) else round
            root = self._fiducial_root_path
        else:
            entity_id = self._bit_ids[bit] if isinstance(bit, int) else bit
            root = self._readouts_root_path
        return self._load_entity_attributes(
            root / tile_id / entity_id, image_names=image_names
        )

    @property
    def datastore_state(self) -> dict | None:
        """Datastore state.

        Returns
        -------
        datastore_state : dict or None
            Datastore state.
        """
        return getattr(self, "_datastore_state", None)

    @datastore_state.setter
    def datastore_state(self, value: dict) -> None:
        """Set the datastore state.

        Parameters
        ----------
        value : dict
            New datastore state.
        """
        state = dict(self.datastore_state or {})
        state.update(value)
        self._save_to_json(state, self._datastore_state_json_path)
        self._datastore_state = state

    def cellpose_pixel_rois_path(self) -> Path:
        """Return the projection-pixel ImageJ ROI archive path."""
        return self._datastore_path / "cell_boundaries/pixel_rois.zip"

    def cellpose_global_rois_path(self) -> Path:
        """Return the global-coordinate ImageJ ROI archive path."""
        return self._datastore_path / "cell_boundaries/cell_rois.zip"

    def cellpose_outlines_path(self) -> Path:
        """Return the Cellpose outlines JSON path."""
        return self._datastore_path / "cell_boundaries/outlines/cell_outlines.json"

    def fused_tiff_path(self) -> Path:
        """Return the optional full-fusion TIFF export directory."""
        return self._datastore_path / "fused_tiff"

    def fiducial_projection_tiff_path(self) -> Path:
        """Return the optional fiducial-projection TIFF export directory."""
        return self._datastore_path / "fiducial_projection_tiff"

    def shading_flatfield_tile_ids_path(self) -> Path:
        """Return the selected illumination-estimation tiles JSON path."""
        return self._datastore_path / "shading/shading_metadata/flatfield_tile_ids.json"

    def _calibrations_attributes_path(self) -> Path:
        """
        Return the acquisition metadata JSON path.

        Returns
        -------
        Path
            Path to acquisition-wide metadata.
        """
        return self._acquisition_metadata_root / "acquisition.json"

    def _load_calibrations_attributes(self) -> dict[str, Any]:
        """
        Load acquisition and calibration metadata from their product folders.

        Returns
        -------
        dict[str, Any]
            Merged metadata loaded from JSON and Parquet files.
        """
        attributes_path = self._calibrations_attributes_path()
        if not attributes_path.is_file():
            raise FileNotFoundError(
                f"Calibration attributes not found: {attributes_path}"
            )
        attributes = self._load_from_json(attributes_path)
        for path in (
            self._datastore_path / "camera/camera_metadata/camera.json",
            self._datastore_path / "shading/shading_metadata/shading.json",
            self._datastore_path / "codewords/normalization/default.json",
        ):
            if path.is_file():
                attributes.update(self._load_from_json(path))
        manifest_path = self._datastore_path / "psfs/psf_metadata/manifest.json"
        if manifest_path.is_file():
            attributes["psf_manifest"] = self._load_from_json(manifest_path)
        for key, path in (
            (
                "exp_order",
                self._datastore_path
                / "acquisition/experiment_order/experiment_order.parquet",
            ),
            (
                "codebook",
                self._datastore_path / "acquisition/codebook/codebook.parquet",
            ),
        ):
            if path.is_file():
                attributes[key] = pd.read_parquet(path).to_numpy().tolist()
        return attributes

    def _set_calibration_attribute(self, key: str, value: Any) -> None:
        """
        Update one calibration metadata field.

        Parameters
        ----------
        key : str
            Calibration metadata key.
        value : Any
            Value to store after JSON conversion.

        Returns
        -------
        None
            No return value.
        """
        if key in {"exp_order", "codebook"}:
            if key == "exp_order":
                path = (
                    self._datastore_path
                    / "acquisition/experiment_order/experiment_order.parquet"
                )
                table = self._experiment_order
            else:
                path = self._datastore_path / "acquisition/codebook/codebook.parquet"
                table = self._codebook
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_to_parquet(table, path)
            return
        if key == "psf_manifest":
            path = self._datastore_path / "psfs/psf_metadata/manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_to_json(self._to_json_compatible(value), path)
            return
        if key in {"camera_model", "e_per_ADU", "noise_map_path"}:
            path = self._datastore_path / "camera/camera_metadata/camera.json"
        elif key == "shading_maps_path":
            path = self._datastore_path / "shading/shading_metadata/shading.json"
        elif key in {
            "global_normalization_vector",
            "global_background_vector",
            "iterative_normalization_vector",
            "iterative_background_vector",
            "decode_normalization_metadata",
        }:
            path = self._datastore_path / "codewords/normalization/default.json"
        else:
            path = self._calibrations_attributes_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        attributes = self._load_from_json(path)
        attributes[str(key)] = self._to_json_compatible(value)
        self._save_to_json(attributes, path)

    def save_chromatic_affine_transforms_zyx_um(
        self,
        calibration: Mapping[str, Any],
    ) -> None:
        """
        Save chromatic affine calibration metadata.

        Parameters
        ----------
        calibration : Mapping[str, Any]
            Calibration metadata containing one 4x4 ``affine_zyx_um`` matrix
            per channel. Each affine maps that channel's physical Z, Y, X
            coordinates onto the lowest-wavelength reference channel.

        Returns
        -------
        None
            Calibration and applicable transforms are stored in the Scene.
        """
        calibration = dict(calibration)
        if "voxel_size_zyx_um" in calibration:
            calibration["voxel_size_zyx_um"] = round_spacing_um(
                calibration["voxel_size_zyx_um"]
            ).tolist()
        from merfish3danalysis.storage.ome import update_chromatic_metadata

        update_chromatic_metadata(
            self._tile_scene_path,
            calibration=self._to_json_compatible(calibration),
        )

    def load_chromatic_affine_transforms_zyx_um(self) -> dict[str, Any]:
        """
        Load chromatic affine calibration metadata.

        Returns
        -------
        dict[str, Any]
            Stored calibration metadata. Returns an empty dictionary when no
            chromatic calibration is present.
        """
        try:
            attributes = open_group(self._tile_scene_path).attrs.get("qi2lab", {})
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {}
        calibration = attributes.get("chromatic_calibration", {})
        if isinstance(calibration, dict):
            if "voxel_size_zyx_um" in calibration:
                calibration["voxel_size_zyx_um"] = round_spacing_um(
                    calibration["voxel_size_zyx_um"]
                ).tolist()
            return calibration
        return {}

    def load_chromatic_affine_transform_zyx_um(
        self,
        channel_name: str | None = None,
        channel_index: int | None = None,
        wavelength_um: float | None = None,
    ) -> np.ndarray:
        """
        Load one chromatic affine transform with identity fallback.

        Parameters
        ----------
        channel_name : str or None, default=None
            Channel name from the calibration metadata.
        channel_index : int or None, default=None
            Channel index from the calibration metadata.
        wavelength_um : float or None, default=None
            Channel wavelength in microns. Used only when channel name/index do
            not find a match.

        Returns
        -------
        numpy.ndarray
            4x4 affine matrix in physical Z, Y, X microns. Identity is returned
            if the calibration or requested channel is absent.
        """
        from merfish3danalysis.utils.decode_warping import select_chromatic_affine

        affine = select_chromatic_affine(
            self.load_chromatic_affine_transforms_zyx_um(),
            channel_name=channel_name,
            channel_index=channel_index,
            wavelength_um=wavelength_um,
        )
        return np.eye(4, dtype=np.float32) if affine is None else affine

    @property
    def microscope_type(self) -> str | None:
        """Microscope type.

        Returns
        -------
        microscope_type : str or None
            Microscope type.
        """
        return getattr(self, "_microscope_type", None)

    @microscope_type.setter
    def microscope_type(self, value: str) -> None:
        """Set the microscope type.

        Parameters
        ----------
        value : str
            New microscope type.
        """
        self._microscope_type = value
        self._set_calibration_attribute("microscope_type", value)

    @property
    def camera_model(self) -> str | None:
        """Camera model.

        Returns
        -------
        camera_model : str or None
            Camera model.
        """
        return getattr(self, "_camera_model", None)

    @camera_model.setter
    def camera_model(self, value: str) -> None:
        """Set the camera model.

        Parameters
        ----------
        value : str
            New camera model.
        """
        self._camera_model = value
        self._set_calibration_attribute("camera_model", value)

    @property
    def num_rounds(self) -> int | None:
        """Number of rounds.

        Returns
        -------
        num_rounds : int
            Number of rounds.
        """
        return getattr(self, "_num_rounds", None)

    @num_rounds.setter
    def num_rounds(self, value: int) -> None:
        """Set the number of rounds.

        Parameters
        ----------
        value : int
            New number of rounds.
        """
        state = open_group(self._tile_scene_path).attrs["qi2lab"]
        if state.get("tile_ids") and value != len(state["experiment_order"]):
            raise ValueError("Number of rounds does not match the tile Scene.")
        self._num_rounds = value
        self._set_calibration_attribute("num_rounds", value)

    @property
    def num_bits(self) -> int:
        """Number of bits.

        Returns
        -------
        num_bits : int
            Number of bits.
        """
        return getattr(self, "_num_bits", None)

    @num_bits.setter
    def num_bits(self, value: int) -> None:
        """Set the number of bits.

        Parameters
        ----------
        value : int
            Number of bits.
        """
        self._num_bits = value
        self._set_calibration_attribute("num_bits", value)

    @property
    def num_tiles(self) -> int | None:
        """Number of tiles.

        Returns
        -------
        num_tiles : int
            Number of tiles.
        """
        return getattr(self, "_num_tiles", None)

    @num_tiles.setter
    def num_tiles(self, value: int) -> None:
        """Set the number of tiles.

        Parameters
        ----------
        value : int
            New number of tiles.
        """
        state = open_group(self._tile_scene_path).attrs["qi2lab"]
        if state.get("tile_ids") and value != len(state["tile_ids"]):
            raise ValueError("Number of tiles does not match the tile Scene.")
        self._num_tiles = value
        self._set_calibration_attribute("num_tiles", value)

        self._tile_ids = []
        for tile_idx in range(value):
            self._tile_ids.append("tile" + str(tile_idx).zfill(4))

    @property
    def channels_in_data(self) -> Collection[str | int] | None:
        """Acquisition channels in image order.

        Returns
        -------
        channels_in_data : Collection[str | int]
            Channel names or indices in acquisition order.
        """
        return getattr(self, "_channels_in_data", None)

    @channels_in_data.setter
    def channels_in_data(self, value: Collection[str | int]) -> None:
        """Set the channels in the data.

        Parameters
        ----------
        value : Collection[str | int]
            Channel names or indices in acquisition order.
        """
        channels = self._to_json_compatible(list(value))
        self._validate_scene_channels(channels)
        self._channels_in_data = channels
        self._set_calibration_attribute("channels_in_data", channels)

    def _validate_scene_channels(self, channels: list[Any] | None) -> None:
        """Keep acquisition channel identities consistent with the Scene."""
        state = open_group(self._tile_scene_path).attrs["qi2lab"]
        if state.get("tile_ids") and state.get("channels_in_data") != channels:
            raise ValueError("Acquisition channels do not match the tile Scene.")

    @property
    def tile_overlap(self) -> float | None:
        """XY tile overlap.

        Returns
        -------
        tile_overlap : float
            XY tile overlap.
        """
        return getattr(self, "_tile_overlap", None)

    @tile_overlap.setter
    def tile_overlap(self, value: float) -> None:
        """Set the tile overlap.

        Parameters
        ----------
        value : float
            New tile overlap.
        """
        self._tile_overlap = value
        self._set_calibration_attribute("tile_overlap", value)

    @property
    def binning(self) -> int | None:
        """Camera binning.

        Returns
        -------
        binning : int
            Camera binning.
        """
        return getattr(self, "_binning", None)

    @binning.setter
    def binning(self, value: int) -> None:
        """Set the camera binning.

        Parameters
        ----------
        value : int
            New camera binning.
        """
        self._binning = value
        self._set_calibration_attribute("binning", value)

    @property
    def e_per_ADU(self) -> float | None:
        """Electrons per camera ADU.

        Returns
        -------
        e_per_ADU : float
            Electrons per camera ADU.
        """
        return getattr(self, "_e_per_ADU", None)

    @e_per_ADU.setter
    def e_per_ADU(self, value: float) -> None:
        """Set the camera conversion (e- per ADU).

        Parameters
        ----------
        value : float
            New camera conversion (e- per ADU).
        """
        self._e_per_ADU = value
        self._set_calibration_attribute("e_per_ADU", value)

    @property
    def na(self) -> float | None:
        """Detection objective numerical aperture (NA).

        Returns
        -------
        na : float
            Detection objective numerical aperture (NA).
        """
        return getattr(self, "_na", None)

    @na.setter
    def na(self, value: float) -> None:
        """Set detection objective numerical aperture (NA).

        Parameters
        ----------
        value: float
            New detection objective numerical aperture (NA)
        """
        self._na = value
        self._set_calibration_attribute("na", value)

    @property
    def ri(self) -> float | None:
        """Detection objective refractive index (RI).

        Returns
        -------
        ri : float
            Detection objective refractive index (RI).
        """
        return getattr(self, "_ri", None)

    @ri.setter
    def ri(self, value: float) -> None:
        """Set detection objective refractive index (RI).

        Parameters
        ----------
        value: float
            New detection objective refractive index (RI)
        """
        self._ri = value
        self._set_calibration_attribute("ri", value)

    @property
    def noise_map(self) -> ArrayLike | None:
        """Camera noise image.

        Returns
        -------
        noise_map : ArrayLike
            Camera noise image.
        """
        if not hasattr(self, "_noise_map"):
            path = self._datastore_path / "camera/camera_noise.ome.zarr"
            if path.exists():
                self._noise_map = self._load_from_zarr_array(path, return_future=False)
        return getattr(self, "_noise_map", None)

    @noise_map.setter
    def noise_map(self, value: ArrayLike) -> None:
        """Set the camera noise image.

        Parameters
        ----------
        value : ArrayLike
            New camera noise image.
        """
        current_local_zarr_path = self._datastore_path / "camera/camera_noise.ome.zarr"

        self._save_to_zarr_array(
            value,
            current_local_zarr_path,
            return_future=False,
            axes="cyx" if np.ndim(value) == 3 else "yx",
            extra_attributes={
                "calibration_type": "camera_noise",
                "channel_indices": list(range(np.shape(value)[0]))
                if np.ndim(value) == 3
                else None,
            },
        )
        self._set_calibration_attribute(
            "noise_map_path", "camera/camera_noise.ome.zarr"
        )
        self._noise_map = value

    @property
    def channel_shading_maps(self) -> ArrayLike | None:
        """Channel shading images.

        Returns
        -------
        channel_shading_maps : ArrayLike or None
            Channel shading images in CYX order, or None when unavailable.
        """
        if not hasattr(self, "_shading_maps"):
            path = self._datastore_path / "shading/shading_maps.ome.zarr"
            if path.exists():
                self._shading_maps = self._load_from_zarr_array(
                    path, return_future=False
                )
        return getattr(self, "_shading_maps", None)

    @channel_shading_maps.setter
    def channel_shading_maps(self, value: ArrayLike) -> None:
        """Set the channel shading images.

        Parameters
        ----------
        value : ArrayLike
            New channel shading images.
        """
        shading_maps = np.asarray(value, dtype=np.float32)
        if shading_maps.ndim == 2:
            shading_maps = np.expand_dims(shading_maps, axis=0)
        if shading_maps.ndim != 3:
            raise ValueError(
                f"Shading maps must be 2D or 3D, got shape {shading_maps.shape}"
            )
        current_local_zarr_path = self._datastore_path / "shading/shading_maps.ome.zarr"

        self._save_to_zarr_array(
            shading_maps,
            current_local_zarr_path,
            return_future=False,
            axes="cyx",
            extra_attributes={
                "calibration_type": "channel_shading",
                "channel_indices": list(range(shading_maps.shape[0])),
            },
        )
        self._set_calibration_attribute(
            "shading_maps_path", "shading/shading_maps.ome.zarr"
        )
        self._shading_maps = shading_maps

    @property
    def channel_psfs(self) -> ArrayLike | None:
        """Channel point spread functions (PSF).

        Returns
        -------
        channel_psfs : ArrayLike or list of ArrayLike or None
            Stacked channel PSFs when shapes match, a list when shapes differ,
            or None when unavailable.
        """
        psfs = getattr(self, "_psfs", None)
        if psfs is None:
            try:
                manifest = self._load_calibrations_attributes().get("psf_manifest", {})
                psf_root = self._datastore_path / "psfs/images"
                if manifest:
                    psfs = [
                        np.asarray(
                            self._load_from_zarr_array(
                                self._image_store_path(psf_root / manifest[key]["id"]),
                                return_future=False,
                            ),
                            dtype=np.float32,
                        )
                        for key in sorted(manifest, key=int)
                    ]
                    self._psfs = psfs
            except (OSError, ValueError, KeyError):
                print("Calibration psfs missing.")
        if psfs is None:
            return None
        if isinstance(psfs, list):
            if len(psfs) == 0:
                return []
            shapes = {tuple(np.asarray(psf).shape) for psf in psfs}
            if len(shapes) == 1:
                return np.stack(psfs, axis=0)
            return psfs
        return psfs

    @channel_psfs.setter
    def channel_psfs(self, value: ArrayLike) -> None:
        """Set the channel point spread functions (PSF).

        Parameters
        ----------
        value : ArrayLike
            New channel point spread functions (PSF).
        """
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                psf_list = [np.asarray(psf, dtype=np.float32) for psf in list(value)]
            elif value.ndim >= 3:
                psf_list = [
                    np.asarray(value[idx], dtype=np.float32)
                    for idx in range(value.shape[0])
                ]
            else:
                psf_list = [np.asarray(value, dtype=np.float32)]
        else:
            psf_list = [np.asarray(psf, dtype=np.float32) for psf in list(value)]

        if len(psf_list) == 0:
            raise ValueError("channel_psfs cannot be empty.")

        psf_root_path = self._datastore_path / "psfs/images"
        psf_root_path.mkdir(exist_ok=True, parents=True)
        psf_manifest: dict[str, Any] = {}

        for psf_idx, psf_array in enumerate(psf_list):
            psf_id = f"psf_{psf_idx:03d}"
            current_psf_path = psf_root_path / Path(psf_id)
            self._save_to_zarr_array(
                psf_array,
                current_psf_path,
                return_future=False,
                extra_attributes={"calibration_type": "psf", "psf_idx": psf_idx},
            )
            psf_manifest[str(psf_idx)] = {
                "id": psf_id,
                "shape_zyx": list(psf_array.shape),
            }

        self._set_calibration_attribute("psf_manifest", psf_manifest)
        self._psfs = psf_list

    @property
    def experiment_order(self) -> pd.DataFrame | None:
        """Round and bit order.

        Returns
        -------
        experiment_order : pd.DataFrame
            Round and bit order.
        """
        experiment_order = getattr(self, "_experiment_order", None)
        if experiment_order is not None:
            return experiment_order

        legacy_experiment_order = getattr(self, "_exp_order", None)
        if legacy_experiment_order is None:
            return None

        self._experiment_order = self._coerce_experiment_order_dataframe(
            legacy_experiment_order
        )
        return self._experiment_order

    def _coerce_experiment_order_dataframe(
        self, value: ArrayLike | pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize experiment order into the canonical DataFrame form.

        Parameters
        ----------
        value : ArrayLike | pd.DataFrame
            Experiment order table or array to normalize.

        Returns
        -------
        pd.DataFrame
            Experiment order as a DataFrame with channel columns.
        """
        if isinstance(value, pd.DataFrame):
            return value

        channel_list = [str(channel) for channel in self._channels_in_data]
        return pd.DataFrame(value, columns=channel_list, dtype="int64")

    @experiment_order.setter
    def experiment_order(self, value: ArrayLike | pd.DataFrame) -> None:
        """Set the round and bit order.

        Parameters
        ----------
        value : ArrayLike or pandas.DataFrame
            New round and bit order.
        """
        order = self._coerce_experiment_order_dataframe(value)
        self._validate_scene_experiment_order(order)
        self._experiment_order = order
        value = order.to_numpy()

        self._set_calibration_attribute("exp_order", self._experiment_order.values)

        if self.num_rounds is None:
            self.num_rounds = int(value[-1, 0])
        else:
            assert self.num_rounds == int(value[-1, 0]), (
                "Number of rounds does not match experiment order file."
            )

        if self.num_bits is None:
            self.num_bits = int(np.max(value[:, 1:]))
        else:
            assert self.num_bits == int(np.max(value[:, 1:])), (
                "Number of bits does not match experiment order file."
            )

        self._round_ids = []
        for round_idx in range(self.num_rounds):
            self._round_ids.append("round" + str(round_idx + 1).zfill(3))

        self._bit_ids = []
        for bit_idx in range(self.num_bits):
            self._bit_ids.append("bit" + str(bit_idx + 1).zfill(3))

    def _validate_scene_experiment_order(self, order: pd.DataFrame) -> None:
        """Keep acquisition channel slots consistent with the initialized Scene."""
        state = open_group(self._tile_scene_path).attrs["qi2lab"]
        if state.get("tile_ids") and not np.array_equal(
            order.to_numpy(), np.asarray(state.get("experiment_order"))
        ):
            raise ValueError("Experiment order does not match the tile Scene.")

    @property
    def codebook(self) -> pd.DataFrame | None:
        """Codebook.

        Returns
        -------
        codebook : pd.DataFrame
            Codebook.
        """
        data = getattr(self, "_codebook", None)

        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy().tolist()
        num_columns = len(data[0]) if data else 0
        columns = ["gene_id"] + [f"bit{i:02d}" for i in range(1, num_columns)]

        return pd.DataFrame(data, columns=columns)

    @codebook.setter
    def codebook(self, value: pd.DataFrame) -> None:
        """Set the codebook.

        Parameters
        ----------
        value : pd.DataFrame
            New codebook.
        """
        self._codebook = value
        self._set_calibration_attribute("codebook", self._codebook.values)

    @property
    def voxel_size_zyx_um(self) -> ArrayLike | None:
        """Voxel size, zyx order (microns).

        Returns
        -------
        voxel_size_zyx_um : ArrayLike
            Voxel size, zyx order (microns).
        """
        value = getattr(self, "_voxel_size_zyx_um", None)
        return None if value is None else self._normalize_voxel_size_zyx_um(value)

    @voxel_size_zyx_um.setter
    def voxel_size_zyx_um(self, value: ArrayLike) -> None:
        """Set the voxel size, zyx order (microns).

        Parameters
        ----------
        value : ArrayLike
            New voxel size, zyx order (microns).
        """
        spacing = self._normalize_voxel_size_zyx_um(value)
        self._set_calibration_attribute("voxel_size_zyx_um", spacing)
        self._voxel_size_zyx_um = spacing

    @staticmethod
    def _normalize_voxel_size_zyx_um(value: ArrayLike) -> list[float]:
        """Validate three positive finite ZYX spacings at stored precision."""
        spacing = round_spacing_um(value)
        if (
            spacing.shape != (3,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError(
                "voxel_size_zyx_um must contain three positive finite ZYX spacings."
            )
        return spacing.tolist()

    @property
    def global_normalization_vector(self) -> ArrayLike | None:
        """Global normalization vector.

        Returns
        -------
        global_normalization_vector : ArrayLike
            Global normalization vector.
        """
        value = getattr(self, "_global_normalization_vector", None)
        if value is None:
            calib_attrs = self._load_calibrations_attributes()

            try:
                value = np.asarray(
                    calib_attrs["global_normalization_vector"], dtype=np.float32
                )
                return value
            except KeyError:
                print("Global normalization vector not calculated.")
                return None
        else:
            return value

    @global_normalization_vector.setter
    def global_normalization_vector(self, value: ArrayLike) -> None:
        """Set the global normalization vector.

        Parameters
        ----------
        value : ArrayLike
            New global normalization vector.
        """
        self._global_normalization_vector = np.asarray(value, dtype=np.float32)
        self._set_calibration_attribute(
            "global_normalization_vector",
            self._global_normalization_vector,
        )

    @property
    def global_background_vector(self) -> ArrayLike | None:
        """Global background vector.

        Returns
        -------
        global_background_vector : ArrayLike
            Global background vector.
        """
        value = getattr(self, "_global_background_vector", None)
        if value is None:
            calib_attrs = self._load_calibrations_attributes()
            try:
                value = np.asarray(
                    calib_attrs["global_background_vector"], dtype=np.float32
                )
                return value
            except KeyError:
                print("Global background vector not calculated.")
                return None
        else:
            return value

    @global_background_vector.setter
    def global_background_vector(self, value: ArrayLike) -> None:
        """Set the global background vector.

        Parameters
        ----------
        value : ArrayLike
            New global background vector.
        """
        self._global_background_vector = np.asarray(value, dtype=np.float32)
        self._set_calibration_attribute(
            "global_background_vector",
            self._global_background_vector,
        )

    @property
    def iterative_normalization_vector(self) -> ArrayLike | None:
        """Iterative normalization vector.

        Returns
        -------
        iterative_normalization_vector : ArrayLike
            Iterative normalization vector.
        """
        value = getattr(self, "_iterative_normalization_vector", None)
        if value is None:
            calib_attrs = self._load_calibrations_attributes()
            try:
                value = np.asarray(
                    calib_attrs["iterative_normalization_vector"], dtype=np.float32
                )
            except KeyError:
                value = None

            if value is None:
                print("Iterative normalization vector not calculated.")
                return None

            return value
        else:
            return value

    @iterative_normalization_vector.setter
    def iterative_normalization_vector(self, value: ArrayLike) -> None:
        """Set the iterative normalization vector.

        Parameters
        ----------
        value : ArrayLike
            New iterative normalization vector.
        """
        self._iterative_normalization_vector = value
        self._set_calibration_attribute(
            "iterative_normalization_vector",
            self._iterative_normalization_vector,
        )

    @property
    def iterative_background_vector(self) -> ArrayLike | None:
        """Iterative background vector.

        Returns
        -------
        iterative_background_vector : ArrayLike
            Iterative background vector.
        """
        value = getattr(self, "_iterative_background_vector", None)
        if value is None:
            calib_attrs = self._load_calibrations_attributes()
            try:
                value = np.asarray(
                    calib_attrs["iterative_background_vector"], dtype=np.float32
                )
            except KeyError:
                value = None
            if value is None:
                print("Iterative background vector not calculated.")
                return None

            return value
        else:
            return value

    @iterative_background_vector.setter
    def iterative_background_vector(self, value: ArrayLike) -> None:
        """Set the iterative background vector.

        Parameters
        ----------
        value : ArrayLike
            New iterative background vector.
        """
        self._iterative_background_vector = value
        self._set_calibration_attribute(
            "iterative_background_vector",
            self._iterative_background_vector,
        )

    @staticmethod
    def _validate_decode_run_key(decode_run_key: str | None) -> str | None:
        """
        Validate an optional decoded-output run key.

        Parameters
        ----------
        decode_run_key : str or None
            Optional subfolder name under decoded-output roots.

        Returns
        -------
        str or None
            Validated run key.
        """
        if decode_run_key is None:
            return None
        decode_run_key = str(decode_run_key)
        if decode_run_key in {".", ".."} or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", decode_run_key
        ):
            raise ValueError(
                "decode_run_key must be a single run name containing only letters, "
                "numbers, '.', '_', and '-'."
            )
        return decode_run_key

    def _decoded_run_root(self, decode_run_key: str | None = None) -> Path:
        """
        Return the local decoded output root for an optional decode run.

        Parameters
        ----------
        decode_run_key : str or None, default None
            Optional decoded-output run key.

        Returns
        -------
        pathlib.Path
            Decoded output root.
        """
        decode_run_key = self._validate_decode_run_key(decode_run_key)
        if decode_run_key is None:
            return self._decoded_root_path
        return self._decoded_root_path / "runs" / decode_run_key

    def _decode_normalization_path(self, decode_run_key: str | None) -> Path:
        """Keep each decode run's normalization metadata with its codewords."""
        root = self._datastore_path / "codewords/normalization"
        key = self._validate_decode_run_key(decode_run_key)
        return root / "default.json" if key is None else root / "runs" / f"{key}.json"

    def decoded_temporary_dir(
        self,
        decode_run_key: str | None = None,
        iteration: int | None = None,
    ) -> Path:
        """
        Return the temporary decoded-output directory for a decode run.

        Parameters
        ----------
        decode_run_key : str or None, default None
            Optional decoded-output run key.
        iteration : int or None, default None
            Optional optimization iteration index.

        Returns
        -------
        pathlib.Path
            Temporary decoded-output directory.
        """
        key = self._validate_decode_run_key(decode_run_key)
        root = self._datastore_path / "codewords/optimization"
        root = root / "default" if key is None else root / "runs" / key
        if iteration is not None:
            root = root / Path(f"iteration_{int(iteration):03d}")
        return root

    def _global_filtered_decoded_root(
        self,
        decode_run_key: str | None = None,
    ) -> Path:
        """
        Return the global filtered decoded-output root for an optional decode run.

        Parameters
        ----------
        decode_run_key : str or None, default None
            Optional decoded-output run key.

        Returns
        -------
        pathlib.Path
            Global filtered decoded-output root.
        """
        root = self._datastore_path / "codewords/global_filtered"
        decode_run_key = self._validate_decode_run_key(decode_run_key)
        if decode_run_key is None:
            return root
        return root / "runs" / decode_run_key

    def load_decode_normalization_vectors(
        self,
        decode_run_key: str | None,
        kind: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Load run-scoped normalization and background vectors.

        Parameters
        ----------
        decode_run_key : str or None
            Optional decoded-output run key. None reads the default vectors.
        kind : {'global', 'iterative'}
            Normalization vector kind.

        Returns
        -------
        tuple[numpy.ndarray or None, numpy.ndarray or None]
            Normalization and background vectors.
        """
        if kind == "global":
            if decode_run_key is None:
                return self.global_normalization_vector, self.global_background_vector
            norm_key = "global_normalization_vector"
            background_key = "global_background_vector"
        elif kind == "iterative":
            if decode_run_key is None:
                return (
                    self.iterative_normalization_vector,
                    self.iterative_background_vector,
                )
            norm_key = "iterative_normalization_vector"
            background_key = "iterative_background_vector"
        else:
            raise ValueError("kind must be one of 'global' or 'iterative'.")

        decode_run_key = self._validate_decode_run_key(decode_run_key)
        run_attrs = self._load_from_json(
            self._decode_normalization_path(decode_run_key)
        )
        normalization_vector = run_attrs.get(norm_key)
        background_vector = run_attrs.get(background_key)
        if normalization_vector is None or background_vector is None:
            return None, None
        return (
            np.asarray(normalization_vector, dtype=np.float32),
            np.asarray(background_vector, dtype=np.float32),
        )

    def save_decode_normalization_vectors(
        self,
        decode_run_key: str | None,
        kind: str,
        normalization_vector: ArrayLike,
        background_vector: ArrayLike,
        decode_mode: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Save default or run-scoped normalization and background vectors.

        Parameters
        ----------
        decode_run_key : str or None
            Optional decoded-output run key. None writes the default vectors.
        kind : {'global', 'iterative'}
            Normalization vector kind.
        normalization_vector : ArrayLike
            Foreground normalization vector.
        background_vector : ArrayLike
            Background vector.
        decode_mode : str or None, default None
            Decode mode metadata.
        metadata : Mapping[str, Any] or None, default None
            Additional provenance for these normalization vectors.
        """
        if kind == "global":
            if decode_run_key is None:
                self.global_normalization_vector = normalization_vector
                self.global_background_vector = background_vector
                self._save_default_decode_normalization_metadata(kind, metadata)
                return
            norm_key = "global_normalization_vector"
            background_key = "global_background_vector"
        elif kind == "iterative":
            if decode_run_key is None:
                self.iterative_normalization_vector = normalization_vector
                self.iterative_background_vector = background_vector
                self._save_default_decode_normalization_metadata(kind, metadata)
                return
            norm_key = "iterative_normalization_vector"
            background_key = "iterative_background_vector"
        else:
            raise ValueError("kind must be one of 'global' or 'iterative'.")

        decode_run_key = self._validate_decode_run_key(decode_run_key)
        path = self._decode_normalization_path(decode_run_key)
        run_attrs = self._load_from_json(path)
        if decode_mode is not None:
            run_attrs["decode_mode"] = str(decode_mode)
        if metadata is not None:
            run_attrs[f"{kind}_metadata"] = dict(metadata)
        run_attrs[norm_key] = np.asarray(normalization_vector, dtype=np.float32)
        run_attrs[background_key] = np.asarray(background_vector, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_to_json(self._to_json_compatible(run_attrs), path)

    def _save_default_decode_normalization_metadata(
        self,
        kind: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        """Save provenance for default, non-run-scoped normalization vectors."""
        if metadata is None:
            return
        path = self._decode_normalization_path(None)
        all_metadata = dict(
            self._load_from_json(path).get("decode_normalization_metadata", {})
        )
        all_metadata[kind] = dict(metadata)
        self._set_calibration_attribute("decode_normalization_metadata", all_metadata)

    def load_decode_normalization_metadata(
        self,
        decode_run_key: str | None,
        kind: str,
    ) -> dict[str, Any] | None:
        """Load provenance for default or run-scoped normalization vectors.

        Parameters
        ----------
        decode_run_key : str or None
            Named decode run, or None for the default run.
        kind : str
            ``"global"`` or ``"iterative"`` normalization.

        Returns
        -------
        dict[str, Any] or None
            Stored provenance, or None when unavailable.
        """
        if kind not in {"global", "iterative"}:
            raise ValueError("kind must be one of 'global' or 'iterative'.")
        calib_attrs = self._load_from_json(
            self._decode_normalization_path(decode_run_key)
        )
        if decode_run_key is None:
            metadata = calib_attrs.get("decode_normalization_metadata", {}).get(kind)
        else:
            decode_run_key = self._validate_decode_run_key(decode_run_key)
            metadata = calib_attrs.get(f"{kind}_metadata")
        return dict(metadata) if isinstance(metadata, Mapping) else None

    @property
    def tile_ids(self) -> Collection[str] | None:
        """Tile IDs.

        Returns
        -------
        tile_ids : Collection[str]
            Tile IDs.
        """
        return getattr(self, "_tile_ids", None)

    @property
    def round_ids(self) -> Collection[str] | None:
        """Round IDs.

        Returns
        -------
        round_ids : Collection[str]
            Round IDs.
        """
        return getattr(self, "_round_ids", None)

    @property
    def bit_ids(self) -> Collection[str] | None:
        """Bit IDs.

        Returns
        -------
        bit_ids : Collection[str]
            Bit IDs.
        """
        return getattr(self, "_bit_ids", None)

    def _init_datastore(self) -> None:
        """Initialize datastore.

        Create directory structure and initialize datastore state.
        """
        from merfish3danalysis.storage.ome import initialize_scene

        self._datastore_path.mkdir(parents=True, exist_ok=False)
        initialize_scene(self._tile_scene_path)
        self._acquisition_metadata_root = (
            self._datastore_path / "acquisition" / "acquisition_metadata"
        )
        self._acquisition_metadata_root.mkdir(parents=True)
        self._save_to_json({}, self._calibrations_attributes_path())
        self.fiducial_folder_name = r"fiducial"
        self._fiducial_root_path = self._tile_scene_path / Path(
            self.fiducial_folder_name
        )
        self._readouts_root_path = self._tile_scene_path / Path(r"readouts")
        self.feature_predictor_folder_name = r"feature_predictor"
        self._feature_predictor_localizations_root_path = (
            self._datastore_path / "ufish" / "localizations"
        )
        self._feature_predictor_localizations_root_path.mkdir(parents=True)
        self._decoded_root_path = self._datastore_path / "codewords" / "tiles"

        self._decoded_root_path.mkdir(parents=True)
        (self._datastore_path / "cell_boundaries").mkdir()

        # initialize datastore state
        self._datastore_state_json_path = (
            self._datastore_path / "datastore_state" / "state.json"
        )
        self._datastore_state_json_path.parent.mkdir()
        self._datastore_state = {
            "Version": 0.6,
            "Initialized": True,
            "Calibrations": False,
            "Corrected": False,
            "LocalRegistered": False,
            "GlobalRegistered": False,
            "Fused": False,
            "SegmentedCells": False,
            "DecodedSpots": False,
            "FilteredSpots": False,
        }

        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @staticmethod
    def _normalize_transform(
        values: Sequence[float] | None, ndim: int, fill: float
    ) -> list[float]:
        """
        Normalize transform vectors to match array dimensionality.

        Parameters
        ----------
        values : Sequence[float] | None
            Optional transform vector.
        ndim : int
            Target array dimensionality.
        fill : float
            Fill value for missing leading dimensions.

        Returns
        -------
        list[float]
            Transform vector with one value per array dimension.
        """
        if values is None:
            return [fill] * ndim
        cast = [float(v) for v in values]
        if len(cast) == ndim:
            return cast
        if len(cast) == 3:
            if ndim == 2:
                return cast[1:]
            if ndim >= 3:
                return [fill] * (ndim - 3) + cast
        raise ValueError(f"Cannot map {len(cast)} transform values onto {ndim} axes.")

    @staticmethod
    def _default_chunks(
        array: np.ndarray,
        spatial_chunk_zyx: tuple[int, int, int] = (16, 512, 512),
    ) -> list[int]:
        """
        Create sane default chunk sizes based on dimensionality.

        Parameters
        ----------
        array : np.ndarray
            Array that will be written.
        spatial_chunk_zyx : tuple[int, int, int], default=(16, 512, 512)
            Desired Z, Y, X storage chunks for local fiducial and readout images.

        Returns
        -------
        list[int]
            Chunk shape matched to the array dimensionality.
        """
        z_chunk, y_chunk, x_chunk = (int(value) for value in spatial_chunk_zyx)
        if array.ndim == 2:
            return [
                min(int(array.shape[0]), y_chunk),
                min(int(array.shape[1]), x_chunk),
            ]
        if array.ndim == 3:
            return [
                min(int(array.shape[0]), z_chunk),
                min(int(array.shape[1]), y_chunk),
                min(int(array.shape[2]), x_chunk),
            ]
        if array.ndim == 4:
            return [
                1,
                min(int(array.shape[1]), z_chunk),
                min(int(array.shape[2]), y_chunk),
                min(int(array.shape[3]), x_chunk),
            ]
        if array.ndim == 5:
            return [
                1,
                1,
                min(int(array.shape[2]), z_chunk),
                min(int(array.shape[3]), y_chunk),
                min(int(array.shape[4]), x_chunk),
            ]
        return list(array.shape)

    @staticmethod
    def _fused_image_chunks(array: np.ndarray) -> list[int]:
        """
        Create chunk sizes tailored for fused image storage.

        Parameters
        ----------
        array : np.ndarray
            Fused image array that will be written.

        Returns
        -------
        list[int]
            Chunk shape for fused image storage.
        """
        shape = [int(dim) for dim in array.shape]
        if array.ndim == 2:
            return [min(shape[0], 2048), min(shape[1], 2048)]
        if array.ndim == 3:
            return [min(shape[0], 16), min(shape[1], 512), min(shape[2], 512)]
        if array.ndim == 4:
            return [
                min(shape[0], 1),
                min(shape[1], 16),
                min(shape[2], 512),
                min(shape[3], 512),
            ]
        if array.ndim == 5:
            return [
                min(shape[0], 1),
                min(shape[1], 1),
                min(shape[2], 16),
                min(shape[3], 512),
                min(shape[4], 512),
            ]
        return list(array.shape)

    @staticmethod
    def _image_store_path(image_path: Path | str) -> Path:
        """
        Normalize image path to the *.ome.zarr folder naming scheme.

        Parameters
        ----------
        image_path : Path | str
            Logical image path or OME-Zarr path.

        Returns
        -------
        Path
            Normalized path ending in ``.ome.zarr``.
        """
        path = Path(image_path)
        if path.name.endswith(".ome.zarr"):
            return path
        if path.name.endswith(".zarr"):
            raise ValueError(
                f"Invalid image store name '{path.name}'. Expected '.ome.zarr' suffix."
            )
        if path.suffixes:
            raise ValueError(
                f"Invalid image store name '{path.name}'. Use bare logical names or '.ome.zarr'."
            )
        return path.with_name(path.name + ".ome.zarr")

    @staticmethod
    def load_image_metadata(image_path: Path | str) -> dict[str, Any]:
        """
        Load extra attributes through yaozarrs.

        Parameters
        ----------
        image_path : Path | str
            Image store path.

        Returns
        -------
        dict[str, Any]
            Extra attributes stored on the image root.
        """
        image_root = qi2labDataStore._image_store_path(image_path)
        attrs = dict(open_group(str(image_root)).attrs)
        attrs.pop("ome", None)
        for key in ("spacing_zyx_um", "voxel_size_zyx_um"):
            if key in attrs:
                attrs[key] = round_spacing_um(attrs[key]).tolist()
        return attrs

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        """
        Convert numpy/scalar containers to JSON-compatible values.

        Parameters
        ----------
        value : Any
            Value to convert before JSON serialization.

        Returns
        -------
        Any
            JSON-compatible representation.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {
                str(k): qi2labDataStore._to_json_compatible(v) for k, v in value.items()
            }
        if isinstance(value, tuple):
            return [qi2labDataStore._to_json_compatible(v) for v in value]
        if isinstance(value, list):
            return [qi2labDataStore._to_json_compatible(v) for v in value]
        return value

    @staticmethod
    def image_shape(image_path: Path | str) -> tuple[int, ...] | None:
        """
        Read image shape without loading all pixels.

        Parameters
        ----------
        image_path : Path | str
            Image store path.

        Returns
        -------
        tuple[int, ...] | None
            Image array shape, or ``None`` when the image is unavailable.
        """
        path = qi2labDataStore._image_store_path(image_path)
        if not path.exists():
            return None

        try:
            group = open_group(str(path))
            dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0]
            array_0 = group[dataset.path]
            shape = array_0.metadata.shape
            return tuple(int(dim) for dim in shape)
        except (FileNotFoundError, KeyError):
            return None

    def _load_entity_attributes(
        self,
        entity_root_path: Path | str,
        image_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Load entity metadata from the Scene and image extra attributes.

        Parameters
        ----------
        entity_root_path : Path | str
            Logical tile-and-round or tile-and-bit path.
        image_names : Sequence[str] | None
            Optional image names whose extra attributes should be merged.

        Returns
        -------
        dict[str, Any]
            Merged entity metadata.
        """
        entity_root = Path(entity_root_path)
        if entity_root.parent.parent == self._fiducial_root_path:
            kind = "fiducial"
        elif entity_root.parent.parent == self._readouts_root_path:
            kind = "readouts"
        else:
            raise ValueError("Expected a tile fiducial or readout entity.")
        merged: dict[str, Any] = {}

        default_images = (
            "corrected_data",
            "decon_data",
            f"{self.feature_predictor_folder_name}_data",
        )
        candidate_names = image_names if image_names is not None else default_images
        for image_name in candidate_names:
            if kind == "fiducial":
                image_path = self.local_image_path(
                    entity_root.parent.name, image_name, round=entity_root.name
                )
            else:
                image_path = self.local_image_path(
                    entity_root.parent.name, image_name, bit=entity_root.name
                )
            if not image_path.exists():
                continue
            merged.update(self.load_image_metadata(image_path))

        state = open_group(self._tile_scene_path).attrs.get("qi2lab", {})
        key = f"{kind}/{entity_root.parent.name}/{entity_root.name}"
        merged.update(state.get("entity_metadata", {}).get(key, {}))

        if kind == "readouts":
            wavelengths = state.get("readout_wavelengths_um", {}).get(
                f"{entity_root.parent.name}/{entity_root.name}"
            )
            if wavelengths is not None:
                merged.update(excitation_um=wavelengths[0], emission_um=wavelengths[1])
            linked_round = state.get("round_links", {}).get(
                f"{entity_root.parent.name}/{entity_root.name}"
            )
            if linked_round is not None:
                merged["round_linker"] = self._round_ids.index(linked_round) + 1

        return merged

    def _save_entity_attributes(
        self,
        entity_root_path: Path | str,
        updates: Mapping[str, Any],
    ) -> None:
        """
        Update entity metadata in the Scene; yaozarrs cannot resume a Scene writer.

        Parameters
        ----------
        entity_root_path : Path | str
            Logical tile-and-round or tile-and-bit path.
        updates : Mapping[str, Any]
            Metadata updates to persist.

        Returns
        -------
        None
            No return value.
        """
        if not updates:
            return

        entity_root = Path(entity_root_path)
        payload = {
            str(k): self._to_json_compatible(v) for k, v in dict(updates).items()
        }

        if entity_root.parent.parent == self._fiducial_root_path:
            kind = "fiducial"
        elif entity_root.parent.parent == self._readouts_root_path:
            kind = "readouts"
        else:
            raise ValueError("Expected a tile fiducial or readout entity.")
        key = f"{kind}/{entity_root.parent.name}/{entity_root.name}"

        from merfish3danalysis.storage._scene_update import update_scene

        def update(
            scene: v06.Scene, attributes: dict[str, Any]
        ) -> tuple[v06.Scene, dict[str, Any]]:
            """Merge the entity update under the Scene lock."""
            state = dict(attributes["qi2lab"])
            metadata = dict(state.get("entity_metadata", {}))
            entity = dict(metadata.get(key, {}))
            entity.update(payload)
            metadata[key] = entity
            state["entity_metadata"] = metadata
            state["generation"] = int(state["generation"]) + 1
            attributes["qi2lab"] = state
            return scene, attributes

        update_scene(self._tile_scene_path, update)

    @staticmethod
    def _normalize_stage_position_zyx_um(stage_zyx_um: ArrayLike) -> np.ndarray:
        """Return ZYX stage coordinates, padding legacy YX positions with Z=0."""
        stage = np.asarray(stage_zyx_um, dtype=np.float32)
        if stage.shape == (2,):
            return np.asarray((0.0, stage[0], stage[1]), dtype=np.float32)
        if stage.shape != (3,):
            raise ValueError(
                "Stage position must contain two (Y, X) or three (Z, Y, X) "
                f"coordinates; got shape {stage.shape}."
            )
        return stage

    def _validate_core_image_shape(
        self,
        entity_root_path: Path | str,
        image_name: str,
        image: ArrayLike,
    ) -> None:
        """
        Enforce corrected/deconvolved/feature image shape consistency.

        Parameters
        ----------
        entity_root_path : Path | str
            Entity folder path.
        image_name : str
            Logical image name being validated.
        image : ArrayLike
            Image data to validate.

        Returns
        -------
        None
            No return value.
        """
        entity_root = Path(entity_root_path)
        shape = tuple(int(v) for v in np.asarray(image).shape)
        required_names = {
            "corrected_data",
            "decon_data",
            f"{self.feature_predictor_folder_name}_data",
        }
        for candidate_name in required_names:
            if candidate_name == image_name:
                continue
            if entity_root.parent.parent == self._fiducial_root_path:
                candidate_path = self.local_image_path(
                    entity_root.parent.name, candidate_name, round=entity_root.name
                )
            elif entity_root.parent.parent == self._readouts_root_path:
                candidate_path = self.local_image_path(
                    entity_root.parent.name, candidate_name, bit=entity_root.name
                )
            else:
                candidate_path = entity_root / candidate_name
            candidate_shape = self.image_shape(candidate_path)
            if candidate_shape is None:
                continue
            if tuple(candidate_shape) != shape:
                raise ValueError(
                    f"Image shape mismatch in {entity_root.name}: "
                    f"{image_name}={shape} but {candidate_name}={candidate_shape}. "
                    "corrected_data, optional decon_data, and "
                    "feature_predictor_data must match."
                )

    @staticmethod
    def _load_from_json(dictionary_path: Path | str) -> dict:
        """Load json as dictionary.

        Parameters
        ----------
        dictionary_path : Path or str
            Path to json file.

        Returns
        -------
        dictionary : dict
            Dictionary from json file.
        """
        try:
            with open(dictionary_path) as f:
                dictionary = json.load(f)
        except FileNotFoundError:
            return {}
        if not isinstance(dictionary, dict):
            raise ValueError(f"Expected a JSON object in {dictionary_path}")
        return dictionary

    @staticmethod
    def _save_to_json(dictionary: dict, dictionary_path: Path | str) -> None:
        """Save dictionary to json.

        Parameters
        ----------
        dictionary : dict
            The data to be saved.
        dictionary_path : Path or str
            The path to the JSON file where the data will be saved.
        """
        serialized = json.dumps(dictionary, indent=4)
        with open(dictionary_path, "w", encoding="utf-8") as file:
            file.write(serialized)

    @staticmethod
    def _load_from_microjson(dictionary_path: Path | str) -> dict:
        """Load cell outlines outlines microjson as dictionary.

        Parameters
        ----------
        dictionary_path : Path or str
            Path to microjson file.

        Returns
        -------
        outlines : dict
            Cell outlines dictionary.
        """
        try:
            with open(dictionary_path) as f:
                data = json.load(f)
                outlines = {}
                for feature in data["features"]:
                    cell_id = feature["properties"]["cell_id"]
                    coordinates = feature["geometry"]["coordinates"][0]
                    outlines[cell_id] = np.array(coordinates)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            outlines = {}
        return outlines

    @staticmethod
    def _check_for_zarr_array(image_path: Path | str) -> None:
        """Check if image exists and is readable via yaozarrs.

        Parameters
        ----------
        image_path : Path or str
            Image store path.
        """
        image_path = qi2labDataStore._image_store_path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        group = open_group(str(image_path))
        dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0]
        _ = group[dataset.path]

    @staticmethod
    def _load_from_zarr_array(
        image_path: Path | str, return_future: bool | None = True
    ) -> "np.ndarray | Future | TensorStore":
        """Read image data via yaozarrs.

        Parameters
        ----------
        image_path : Path or str
            Image store path.
        return_future : bool or None
            Return read future (True), immediately read (False), or return the
            TensorStore array without issuing a read (None).

        Returns
        -------
        array : tensorstore.Future or numpy.ndarray or tensorstore.TensorStore
            Read future, immediate array, or lazy TensorStore handle obtained
            through yaozarrs.
        """
        image_path = qi2labDataStore._image_store_path(image_path)
        group = open_group(str(image_path))
        dataset = group.ome_metadata(strict=True).multiscales[0].datasets[0]
        current_array = group[dataset.path].to_tensorstore()

        if return_future is None:
            return current_array
        read_future = current_array.read()
        return read_future if return_future else read_future.result()

    @staticmethod
    def _save_to_zarr_array(
        array: ArrayLike,
        image_path: Path | str,
        return_future: bool | None = False,
        *,
        chunks: Sequence[int] | None = None,
        shards: Sequence[int] | None = None,
        ome_scale: Sequence[float] | None = None,
        ome_translation: Sequence[float] | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
        axes: str | None = None,
        scene_path: Path | None = None,
        scene_selection: str | None = None,
        scene_frame: str | None = None,
        scene_affine: ArrayLike | None = None,
    ) -> "WriteFutures | None":
        """Save image data using yaozarrs and its TensorStore backend.

        Parameters
        ----------
        array : ArrayLike
            Array to save.
        image_path : Path or str
            Image store path.
        chunks : Sequence[int] or None
            Storage chunk shape. None selects the datastore default.
        shards : Sequence[int] or None
            Storage shard shape, passed to yaozarrs; None disables sharding.
        ome_scale : Sequence[float] or None
            Pixel spacing in microns, in image-axis order or ZYX order.
        ome_translation : Sequence[float] or None
            Image origin in microns, in image-axis order or ZYX order.
        extra_attributes : Mapping or None
            Additional image metadata.
        axes : str or None
            Array axes when the default spatial layout does not apply.
        return_future : bool or None
            Return the write-completion handle (True), or wait for the write
            to finish and return None (False or None).
        scene_path, scene_selection, scene_frame : Path or str or None
            Publish a native image in the Scene, when all three are supplied.
        scene_affine : ArrayLike or None
            Optional physical ZYX affine from the image to its Scene frame.

        Returns
        -------
        write_future : tensorstore.WriteFutures or None
            Pixel-write completion handle when return_future is True.

        Raises
        ------
        OSError, ValueError
            Storage or metadata errors. Failed writes are not reported as success.
        """
        image_path = qi2labDataStore._image_store_path(image_path)

        image_array = np.asarray(array)
        if image_array.dtype == np.float64:
            image_array = image_array.astype(np.float32)

        if image_array.ndim < 2 or image_array.ndim > 5:
            raise ValueError(
                f"Unsupported array ndim for image write: {image_array.ndim}"
            )

        if chunks is None:
            chunks = qi2labDataStore._default_chunks(image_array)
        scale = qi2labDataStore._normalize_transform(ome_scale, image_array.ndim, 1.0)
        translation = qi2labDataStore._normalize_transform(
            ome_translation, image_array.ndim, 0.0
        )

        if scene_path is not None:
            from merfish3danalysis.storage.ome import publish_image

            if not scene_selection or not scene_frame:
                raise ValueError("A Scene image needs a selection and a frame.")
            axis_names = axes if axes is not None else "tczyx"[-image_array.ndim :]
            if len(axis_names) != image_array.ndim:
                raise ValueError(
                    "The number of axes must match the array dimensionality."
                )
            model = v06.Image(
                multiscales=[
                    v06.Multiscale.from_dims(
                        [
                            DimSpec(
                                name=axis,
                                size=size,
                                scale=round_pixel_size_um(step)
                                if axis in "zyx"
                                else step,
                                translation=offset,
                                unit="micrometer"
                                if axis in "zyx"
                                else ("second" if axis == "t" else None),
                            )
                            for axis, size, step, offset in zip(
                                axis_names,
                                image_array.shape,
                                scale,
                                translation,
                                strict=True,
                            )
                        ]
                    )
                ]
            )
            products = open_group(scene_path).attrs["qi2lab"]["products"]
            previous = products.get(scene_selection)
            published = publish_image(
                scene_path,
                scene_selection,
                scene_frame,
                model,
                image_array,
                expected_revision=None if previous is None else previous["revision"],
                chunks=chunks,
                shards=shards,
                extra_attributes=extra_attributes,
                return_write_future=bool(return_future),
                affine_zyx_um=scene_affine,
                missing_z_origin_um=(
                    float(ome_translation[-3])
                    if image_array.ndim == 2
                    and ome_translation is not None
                    and len(ome_translation) >= 3
                    else 0
                ),
            )
            return published[1] if return_future else None

        image_path.parent.mkdir(parents=True, exist_ok=True)
        axis_names = axes if axes is not None else "tczyx"[-image_array.ndim :]
        if len(axis_names) != image_array.ndim:
            raise ValueError("The number of axes must match the array dimensionality.")
        image_metadata = v06.Image(
            multiscales=[
                v06.Multiscale.from_dims(
                    [
                        DimSpec(
                            name=axis,
                            size=size,
                            scale=round_pixel_size_um(step) if axis in "zyx" else step,
                            translation=offset,
                            unit="micrometer"
                            if axis in "zyx"
                            else ("second" if axis == "t" else None),
                        )
                        for axis, size, step, offset in zip(
                            axis_names,
                            image_array.shape,
                            scale,
                            translation,
                            strict=True,
                        )
                    ]
                )
            ]
        )

        chunk_spec = tuple(int(c) for c in chunks)
        write_options = {
            "extra_attributes": qi2labDataStore._to_json_compatible(extra_attributes)
            if extra_attributes
            else None,
            "writer": "tensorstore",
            "overwrite": True,
            "chunks": chunk_spec,
            "shards": tuple(shards) if shards is not None else None,
            "compression": "blosc-zstd",
        }
        if return_future:
            from yaozarrs.write.v06 import prepare_image

            # yaozarrs creates the complete OME-Zarr v3 structure. Its write_image
            # API is synchronous; writing the prepared array is necessary to
            # preserve the datastore's TensorStore write-completion handle.
            _, arrays = prepare_image(
                dest=str(image_path),
                image=image_metadata,
                datasets=(image_array.shape, image_array.dtype),
                **write_options,
            )
            return arrays[image_metadata.multiscales[0].datasets[0].path].write(
                image_array
            )
        write_image(
            dest=str(image_path),
            image=image_metadata,
            datasets=image_array,
            **write_options,
        )
        return None

    @staticmethod
    def _load_from_parquet(parquet_path: Path | str) -> pd.DataFrame:
        """Load dataframe from parquet.

        Parameters
        ----------
        parquet_path : Path or str
            Path to parquet file.

        Returns
        -------
        df : pd.DataFrame
            Dataframe from parquet file.
        """
        return pd.read_parquet(parquet_path)

    @staticmethod
    def _save_to_parquet(df: pd.DataFrame, parquet_path: Path | str) -> None:
        """Save dataframe to parquet.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        parquet_path : Path or str
            Path to parquet file.
        """
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

    @staticmethod
    def _save_to_csv_gz(df: pd.DataFrame, csv_gz_path: Path | str) -> None:
        """Save dataframe to gzipped CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        csv_gz_path : Path or str
            Path to gzipped CSV file.
        """
        df.to_csv(csv_gz_path, index=False, compression="gzip")

    def _parse_datastore(self, validate: bool = True) -> None:
        """
        Parse datastore to discover available components.

        Parameters
        ----------
        validate : bool
            Whether to validate existing zarr arrays while parsing.

        Returns
        -------
        None
            No return value.
        """
        if not self._tile_scene_path.is_dir():
            raise ValueError(
                "This datastore does not contain tiles.ome.zarr; regenerate it "
                "from the acquisition data for the current layout."
            )
        if not isinstance(
            open_group(self._tile_scene_path).ome_metadata(strict=True), v06.Scene
        ):
            raise ValueError("tiles.ome.zarr must contain an OME-NGFF 0.6 Scene.")

        # directory structure as defined by qi2lab spec
        self._datastore_state_json_path = (
            self._datastore_path / "datastore_state" / "state.json"
        )
        # Read the processing state from its datastore product folder.
        with open(self._datastore_state_json_path) as json_file:
            self._datastore_state = json.load(json_file)
        if float(self._datastore_state["Version"]) != 0.6:
            raise ValueError("Only datastore version 0.6 is supported by this build.")

        self.fiducial_folder_name = "fiducial"
        self.feature_predictor_folder_name = "feature_predictor"
        self._fiducial_root_path = self._tile_scene_path / Path(
            self.fiducial_folder_name
        )
        self._feature_predictor_localizations_root_path = (
            self._datastore_path / "ufish" / "localizations"
        )
        self._acquisition_metadata_root = (
            self._datastore_path / "acquisition" / "acquisition_metadata"
        )
        self._readouts_root_path = self._tile_scene_path / Path(r"readouts")
        self._decoded_root_path = self._datastore_path / "codewords" / "tiles"

        order_path = (
            self._datastore_path
            / "acquisition/experiment_order/experiment_order.parquet"
        )
        scene_state = open_group(self._tile_scene_path).attrs["qi2lab"]
        if scene_state.get("tile_ids"):
            if not order_path.is_file():
                raise ValueError("Experiment order is missing for the tile Scene.")
            self._validate_scene_experiment_order(pd.read_parquet(order_path))
            attributes_path = self._calibrations_attributes_path()
            channels = (
                self._load_from_json(attributes_path).get("channels_in_data")
                if attributes_path.is_file()
                else None
            )
            self._validate_scene_channels(channels)

        # validate calibrations
        if self._datastore_state["Calibrations"]:
            if not (self._acquisition_metadata_root.exists()):
                raise FileNotFoundError("Calibration data directory is missing.")
            attributes = self._load_calibrations_attributes()

            keys_to_check = [
                "num_rounds",
                "num_tiles",
                "channels_in_data",
                "tile_overlap",
                "binning",
                "e_per_ADU",
                "na",
                "ri",
                "exp_order",
                "codebook",
                "num_bits",
                "microscope_type",
                "camera_model",
                "voxel_size_zyx_um",
            ]
            for key in keys_to_check:
                if key not in attributes.keys():
                    raise KeyError("Calibration attributes incomplete")

            self._num_rounds = attributes["num_rounds"]
            self._num_tiles = attributes["num_tiles"]
            self._channels_in_data = attributes["channels_in_data"]
            self._tile_overlap = attributes["tile_overlap"]
            self._binning = attributes["binning"]
            self._e_per_ADU = attributes["e_per_ADU"]
            self._na = attributes["na"]
            self._ri = attributes["ri"]
            self._exp_order = attributes["exp_order"]
            self._codebook = attributes["codebook"]
            self._num_bits = attributes["num_bits"]
            self._microscope_type = attributes["microscope_type"]
            self._camera_model = attributes["camera_model"]
            self._voxel_size_zyx_um = self._normalize_voxel_size_zyx_um(
                attributes["voxel_size_zyx_um"]
            )

            if getattr(self, "_exp_order", None) is not None:
                self._experiment_order = self._coerce_experiment_order_dataframe(
                    self._exp_order
                )

            if getattr(self, "_num_tiles", None) is not None:
                self._tile_ids = [
                    "tile" + str(tile_idx).zfill(4)
                    for tile_idx in range(self._num_tiles)
                ]
            if getattr(self, "_num_rounds", None) is not None:
                self._round_ids = [
                    "round" + str(round_idx + 1).zfill(3)
                    for round_idx in range(self._num_rounds)
                ]
            if getattr(self, "_num_bits", None) is not None:
                self._bit_ids = [
                    "bit" + str(bit_idx + 1).zfill(3)
                    for bit_idx in range(self._num_bits)
                ]

            if validate:
                _ = self.channel_psfs

        # validate fiducial and readout bits data
        if self._datastore_state["Corrected"] and validate:
            state = open_group(self._tile_scene_path).attrs["qi2lab"]
            bits_by_round = state.get("bits_by_round", {})
            if (
                state.get("tile_ids") != self._tile_ids
                or set(bits_by_round) != set(self._round_ids)
                or {bit for bits in bits_by_round.values() for bit in bits}
                != set(self._bit_ids)
            ):
                raise ValueError("Datastore acquisition does not match the tile Scene.")

            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
                attributes = self._load_entity_attributes(entity_root)

                keys_to_check = [
                    "excitation_um",
                    "emission_um",
                    # "exposure_ms",
                    "psf_idx",
                ]

                for key in keys_to_check:
                    if key not in attributes.keys():
                        print(tile_id, round_id, key)
                        raise KeyError("Corrected fiducial attributes incomplete")
                if self.load_local_stage_position_zyx_um(tile_id, round_id) is None:
                    raise KeyError("Corrected fiducial stage transforms incomplete")
                if "bit_linker" not in attributes:
                    print(tile_id, round_id, "bit_linker")
                    raise KeyError("Corrected fiducial attributes incomplete")

                current_local_zarr_path = self.local_image_path(
                    tile_id, "corrected_data", round=round_id
                )

                try:
                    self._check_for_zarr_array(current_local_zarr_path)
                except (OSError, ValueError):
                    print(tile_id, round_id)
                    print("Corrected fiducial data missing.")

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                entity_root = self._readouts_root_path / Path(tile_id) / Path(bit_id)
                attributes = self._load_entity_attributes(entity_root)

                keys_to_check = [
                    "excitation_um",
                    "emission_um",
                    # "exposure_ms",
                    "psf_idx",
                ]
                for key in keys_to_check:
                    if key not in attributes.keys():
                        raise KeyError("Corrected readout attributes incomplete")
                if "round_linker" not in attributes:
                    raise KeyError("Corrected readout attributes incomplete")

                current_local_zarr_path = self.local_image_path(
                    tile_id, "corrected_data", bit=bit_id
                )

                try:
                    self._check_for_zarr_array(current_local_zarr_path)
                except (OSError, ValueError):
                    print(tile_id, bit_id)
                    print("Corrected readout data missing.")

        # check and validate local transform and preprocessing data
        if self._datastore_state["LocalRegistered"] and validate:
            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                if round_id != self._round_ids[0]:
                    if (
                        self.load_local_round_transform_zyx_um(tile_id, round_id)
                        is None
                    ):
                        raise KeyError(
                            f"{round_id, tile_id} local round transform missing"
                        )

                corrected_shape = self.image_shape(
                    self.local_image_path(tile_id, "corrected_data", round=round_id)
                )
                decon_shape = self.image_shape(
                    self.local_image_path(tile_id, "decon_data", round=round_id)
                )
                if (
                    corrected_shape is not None
                    and decon_shape is not None
                    and corrected_shape != decon_shape
                ):
                    raise ValueError(
                        f"{tile_id} {round_id} corrected and decon shapes differ: "
                        f"{corrected_shape} != {decon_shape}"
                    )

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                current_local_zarr_path = self.local_image_path(
                    tile_id, f"{self.feature_predictor_folder_name}_data", bit=bit_id
                )

                try:
                    self._check_for_zarr_array(current_local_zarr_path)
                except (OSError, ValueError):
                    print(tile_id, bit_id)
                    print("feature_predictor prediction missing.")
                corrected_shape = self.image_shape(
                    self.local_image_path(tile_id, "corrected_data", bit=bit_id)
                )
                decon_shape = self.image_shape(
                    self.local_image_path(tile_id, "decon_data", bit=bit_id)
                )
                feature_shape = self.image_shape(current_local_zarr_path)
                shapes = [
                    shape
                    for shape in (corrected_shape, decon_shape, feature_shape)
                    if shape is not None
                ]
                if len(shapes) > 1 and any(shape != shapes[0] for shape in shapes[1:]):
                    raise ValueError(
                        f"{tile_id} {bit_id} corrected/decon/feature image shapes differ: "
                        f"{corrected_shape}, {decon_shape}, {feature_shape}"
                    )

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                current_feature_predictor_path = (
                    self._feature_predictor_localizations_root_path
                    / Path(tile_id)
                    / Path(bit_id + ".parquet")
                )
                if not (current_feature_predictor_path.exists()):
                    raise FileNotFoundError(
                        tile_id
                        + " "
                        + bit_id
                        + " feature_predictor localization missing"
                    )

        # check and validate global registered data
        if self._datastore_state["GlobalRegistered"] and validate:
            for tile_id in self._tile_ids:
                coordinates = self.load_global_coord_xforms_um(tile_id)
                if coordinates is None or any(value is None for value in coordinates):
                    raise KeyError("Global registration missing")

        # check and validate fused
        if self._datastore_state["Fused"] and validate:
            fused_image_path = self.fused_image_path()
            attributes = self.load_image_metadata(fused_image_path)

            keys_to_check = ["affine_zyx_um", "origin_zyx_um", "spacing_zyx_um"]

            for key in keys_to_check:
                if key not in attributes.keys():
                    raise KeyError("Fused image metadata missing")

            current_local_zarr_path = str(fused_image_path)

            try:
                self._check_for_zarr_array(current_local_zarr_path)
            except (OSError, ValueError) as exc:
                raise FileNotFoundError("Fused data missing.") from exc
            if self.load_global_fiducial_projection(return_future=None) is None:
                raise FileNotFoundError("Fiducial projection missing.")

        # check and validate cellpose segmentation
        if self._datastore_state["SegmentedCells"] and validate:
            if self.load_global_cellpose_segmentation_image(return_future=None) is None:
                raise FileNotFoundError("Cellpose labels missing.")

            cell_outlines_path = self.cellpose_global_rois_path()
            if not (cell_outlines_path.exists()):
                raise FileNotFoundError("Cellpose cell outlines missing.")

        # check and validate decoded spots
        if self._datastore_state["DecodedSpots"] and validate:
            roots = [self._decoded_run_root()]
            runs = self._decoded_root_path / "runs"
            if runs.is_dir():
                roots.extend(path for path in runs.iterdir() if path.is_dir())
            if not any(
                all(
                    (root / f"{tile_id}.parquet").is_file()
                    for tile_id in self._tile_ids
                )
                for root in roots
            ):
                raise FileNotFoundError("Decoded spots missing for a complete run.")

        # check and validate filtered decoded spots
        if self._datastore_state["FilteredSpots"] and validate:
            root = self._global_filtered_decoded_root()
            runs = root / "runs"
            roots = [root]
            if runs.is_dir():
                roots.extend(path for path in runs.iterdir() if path.is_dir())
            if not any((path / "decoded_features.parquet").is_file() for path in roots):
                raise FileNotFoundError("filtered decoded spots missing.")

    def load_codebook_parsed(
        self,
    ) -> tuple[Collection[str], ArrayLike] | None:
        """Load and split codebook into gene_ids and codebook matrix.

        Returns
        -------
        gene_ids : Collection[str]
            Gene IDs.
        codebook_matrix : ArrayLike
            Codebook matrix.
        """
        try:
            codebook_df = self.codebook
            if codebook_df is None:
                return None
            gene_ids = codebook_df.iloc[:, 0].tolist()
            codebook_matrix = codebook_df.iloc[:, 1:].to_numpy().astype(int)
            return gene_ids, codebook_matrix
        except (KeyError, ValueError, TypeError):
            print("Error parsing codebook.")
            return None

    def initialize_tile(
        self,
        tile: int | str,
    ) -> None:
        """Initialize a tile's acquisition relationships in the Scene.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        """
        if getattr(self, "_experiment_order", None) is None:
            print("Assign experimental order before creating tiles.")
            return None

        if getattr(self, "_num_tiles", None) is None:
            print("Assign number of tiles before creating tiles.")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tile id.")
                return None
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if not getattr(self, "_scene_acquisition_initialized", False):
            from merfish3danalysis.storage.ome import (
                ConcurrentSceneUpdateError,
                publish_acquisition_scene,
            )

            bits_by_round = {
                round_id: [self._bit_ids[int(bit) - 1] for bit in row[1:] if bit > 0]
                for round_id, row in zip(
                    self._round_ids, self._experiment_order.to_numpy(), strict=True
                )
            }
            bit_linkers_by_round = {
                round_id: [int(bit) for bit in row[1:]]
                for round_id, row in zip(
                    self._round_ids, self._experiment_order.to_numpy(), strict=True
                )
            }
            state = open_group(self._tile_scene_path).attrs["qi2lab"]
            if not state.get("tile_ids"):
                try:
                    publish_acquisition_scene(
                        self._tile_scene_path,
                        self._tile_ids,
                        bits_by_round,
                        bit_linkers_by_round,
                        self._experiment_order.to_numpy().tolist(),
                        self._to_json_compatible(list(self._channels_in_data)),
                    )
                except ConcurrentSceneUpdateError:
                    pass
                state = open_group(self._tile_scene_path).attrs["qi2lab"]
            if (
                state["tile_ids"] != self._tile_ids
                or state["bits_by_round"] != bits_by_round
                or state.get("experiment_order")
                != self._experiment_order.to_numpy().tolist()
                or state.get("channels_in_data")
                != self._to_json_compatible(list(self._channels_in_data))
            ):
                raise ValueError("Datastore acquisition does not match the Scene.")
            self._scene_acquisition_initialized = True

    def load_local_bit_linker(
        self,
        tile: int | str,
        round: int | str,
    ) -> Sequence[int] | None:
        """Load readout bits linked to fiducial round for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.

        Returns
        -------
        bit_linker : Sequence[int] or None
            Readout bits linked to fiducial round for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id.")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            attributes = self._load_entity_attributes(entity_root)
            bit_linker = attributes.get("bit_linker")
            if bit_linker is None:
                print(tile_id, round_id)
                print("Bit linker attribute not found.")
                return None
            return [int(v) for v in list(bit_linker)]
        except (TypeError, ValueError):
            print(tile_id, round_id)
            print("Bit linker attribute not found.")
            return None

    def save_local_bit_linker(
        self,
        bit_linker: Sequence[int],
        tile: int | str,
        round: int | str,
    ) -> None:
        """Save readout bits linked to fiducial round for one tile.

        Parameters
        ----------
        bit_linker : Sequence[int]
            Readout bits linked to fiducial round for one tile.
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id.")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            values = [int(v) for v in list(bit_linker)]
            self._save_entity_attributes(
                entity_root_path=entity_root, updates={"bit_linker": values}
            )
        except (TypeError, ValueError):
            print(tile_id, round_id)
            print("Error writing bit linker attribute.")
            return None

    def load_local_round_linker(
        self,
        tile: int | str,
        bit: int | str,
    ) -> Sequence[int] | None:
        """Load fiducial round linked to readout bit for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.

        Returns
        -------
        round_linker : Sequence[int] or None
            Fiducial round linked to readout bit for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id.")
                return None
            else:
                bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        try:
            entity_root = self._readouts_root_path / Path(tile_id) / Path(bit_id)
            attributes = self._load_entity_attributes(entity_root)
            round_linker = attributes.get("round_linker")
            if round_linker is None:
                print(tile_id, bit_id)
                print("Round linker attribute not found.")
                return None
            return int(round_linker)
        except (TypeError, ValueError):
            print(tile_id, bit_id)
            print("Round linker attribute not found.")
            return None

    def save_local_round_linker(
        self,
        round_linker: int,
        tile: int | str,
        bit: int | str,
    ) -> None:
        """Save fiducial round linker attribute to readout bit for one tile.

        Parameters
        ----------
        round_linker : int
            Fiducial round linked to readout bit for one tile.
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id.")
                return None
            else:
                bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        try:
            round_index = int(round_linker) - 1
            if not 0 <= round_index < len(self._round_ids):
                raise ValueError("Invalid linked round.")
        except (TypeError, ValueError):
            print(tile_id, bit_id)
            print("Error writing round linker attribute.")
            return None
        from merfish3danalysis.storage._scene_update import update_scene

        target_round = self._round_ids[round_index]
        target_native = f"{tile_id}/{target_round}/{bit_id}/native"
        target_chromatic = f"{tile_id}/{target_round}/{bit_id}/chromatic"

        def update(
            scene: v06.Scene, attributes: dict[str, Any]
        ) -> tuple[v06.Scene, dict[str, Any]]:
            """Move the bit's image and chromatic links in one Scene update."""
            state = dict(attributes["qi2lab"])
            transforms = list(scene.scene.coordinateTransformations)
            chromatic = next(
                t
                for t in transforms
                if t.name
                and t.name.startswith(f"{tile_id}/")
                and t.name.endswith(f"/{bit_id}/chromatic")
            )
            old_native = chromatic.input.name
            old_name = chromatic.name
            if old_name == target_chromatic:
                return scene, attributes
            systems = [
                system.model_copy(update={"name": target_native})
                if system.name == old_native
                else system
                for system in scene.scene.coordinateSystems
            ]
            new_transforms = []
            for transform in transforms:
                if transform.name == old_name:
                    transform = transform.model_copy(
                        update={
                            "name": target_chromatic,
                            "input": v06.CoordinateSystemRef(name=target_native),
                            "output": v06.CoordinateSystemRef(
                                name=f"{tile_id}/{target_round}/moving"
                            ),
                        }
                    )
                elif (
                    transform.output.name == old_native
                    and transform.output.path is None
                ):
                    transform = transform.model_copy(
                        update={"output": v06.CoordinateSystemRef(name=target_native)}
                    )
                new_transforms.append(transform)
            for key in (
                "transform_status",
                "transform_versions",
                "transform_provenance",
            ):
                values = dict(state.get(key, {}))
                if old_name in values:
                    values[target_chromatic] = values.pop(old_name)
                    if key == "transform_versions":
                        values[target_chromatic] += 1
                state[key] = values
            products = {
                name: {**product, "frame": target_native}
                if product.get("frame") == old_native
                else product
                for name, product in state["products"].items()
            }
            links = dict(state.get("round_links", {}))
            links[f"{tile_id}/{bit_id}"] = target_round
            state.update(
                products=products,
                round_links=links,
                generation=int(state["generation"]) + 1,
            )
            attributes["qi2lab"] = state
            return scene.model_copy(
                update={
                    "scene": scene.scene.model_copy(
                        update={
                            "coordinateSystems": systems,
                            "coordinateTransformations": new_transforms,
                        }
                    )
                }
            ), attributes

        update_scene(self._tile_scene_path, update)

    def load_local_stage_position_zyx_um(
        self,
        tile: int | str,
        round: int | str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Load tile stage position for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.

        Returns
        -------
        stage_zyx_um : ArrayLike or None
            Tile stage position in Z, Y, X microns. Legacy two-coordinate Y, X
            metadata is returned with Z=0 without modifying the stored data.
        affine_zyx_um: ArrayLike or None
            Affine transformation between stage and camera
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id.")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        group = open_group(self._tile_scene_path)
        scene = group.ome_metadata(strict=True)
        names = [f"{tile_id}/{round_id}/stage", f"{tile_id}/{round_id}/camera"]
        statuses = group.attrs["qi2lab"]["transform_status"]
        if any(statuses.get(name) != "calculated" for name in names):
            return None
        transforms = {
            transform.name: transform
            for transform in scene.scene.coordinateTransformations
        }
        stage, camera = (transforms[name] for name in names)
        if not isinstance(stage, v06.TranslationTransformation) or not isinstance(
            camera, v06.AffineTransformation
        ):
            raise ValueError("Stage and camera transforms have unexpected NGFF types.")
        affine = np.eye(4, dtype=np.float32)
        affine[:3] = np.asarray(camera.affine, dtype=np.float32)
        return np.asarray(stage.translation, dtype=np.float32), affine

    def save_local_stage_position_zyx_um(
        self,
        stage_zyx_um: ArrayLike,
        affine_zyx_px: ArrayLike,
        tile: int | str,
        round: int | str,
    ) -> None:
        """Save tile stage position for one tile.

        Parameters
        ----------
        stage_zyx_um : ArrayLike
            Tile stage position in Z, Y, X microns. Two-coordinate Y, X input
            is accepted and stored as Z, Y, X with Z=0.
        affine_zyx_px: ArrayLike
            4x4 homogeneous affine matrix for stage transformation
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id.")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        from merfish3danalysis.storage.ome import publish_transforms

        stage_zyx_um = self._normalize_stage_position_zyx_um(stage_zyx_um)
        camera = np.asarray(affine_zyx_px, dtype=np.float32)
        if (
            camera.shape != (4, 4)
            or not np.isfinite(stage_zyx_um).all()
            or not np.isfinite(camera).all()
            or not np.array_equal(camera[3], [0, 0, 0, 1])
        ):
            raise ValueError(
                "Stage coordinates and the 4x4 camera affine must be valid."
            )
        group = open_group(self._tile_scene_path)
        scene = group.ome_metadata(strict=True)
        names = [f"{tile_id}/{round_id}/stage", f"{tile_id}/{round_id}/camera"]
        transforms = {
            transform.name: transform
            for transform in scene.scene.coordinateTransformations
        }
        expected = [transforms[name] for name in names]
        replacements = [
            v06.TranslationTransformation(
                name=names[0],
                input=expected[0].input,
                output=expected[0].output,
                translation=stage_zyx_um.tolist(),
            ),
            v06.AffineTransformation(
                name=names[1],
                input=expected[1].input,
                output=expected[1].output,
                affine=camera[:3].tolist(),
            ),
        ]
        state = group.attrs["qi2lab"]
        publish_transforms(
            self._tile_scene_path,
            expected,
            replacements,
            expected_statuses={name: state["transform_status"][name] for name in names},
            expected_revisions={
                name: state["transform_versions"][name] for name in names
            },
        )

    def load_local_wavelengths_um(
        self,
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
    ) -> tuple[float, float] | None:
        """Load wavelengths for fiducial OR readout bit for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str or None
            Round index or round id.
        bit : int or str or None
            Bit index or bit id.

        Returns
        -------
        wavelengths_um : tuple[float, float] or None
            Wavelengths for fiducial OR readout bit for one tile.
        """
        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if bit is not None:
            if isinstance(bit, int):
                if bit < 0 or bit >= len(self._bit_ids):
                    print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                    return None
                else:
                    local_id = self._bit_ids[bit]
            elif isinstance(bit, str):
                if bit not in self._bit_ids:
                    print("Set valid bit id")
                    return None
                else:
                    local_id = bit
            else:
                print("'bit' must be integer index or string identifier")
                return None
            entity_root = self._readouts_root_path / Path(tile_id) / Path(local_id)
        else:
            if isinstance(round, int):
                if round < 0 or round >= len(self._round_ids):
                    print("Set round index >=0 and <" + str(self._num_rounds))
                    return None
                else:
                    local_id = self._round_ids[round]
            elif isinstance(round, str):
                if round not in self._round_ids:
                    print("Set valid round id")
                    return None
                else:
                    local_id = round
            else:
                print("'round' must be integer index or string identifier")
                return None
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(local_id)

        try:
            if bit is not None:
                state = open_group(self._tile_scene_path).attrs.get("qi2lab", {})
                values = state.get("readout_wavelengths_um", {})[
                    f"{tile_id}/{local_id}"
                ]
                return (values[0], values[1])
            attributes = self._load_entity_attributes(entity_root)
            ex_wavelength_um = attributes["excitation_um"]
            em_wavelength_um = attributes["emission_um"]
            return (ex_wavelength_um, em_wavelength_um)
        except KeyError:
            print("Wavelength attributes not found.")
            return None

    def save_local_wavelengths_um(
        self,
        wavelengths_um: tuple[float, float],
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
    ) -> tuple[float, float] | None:
        """Save wavelengths for fiducial OR readout bit for one tile.

        Parameters
        ----------
        wavelengths_um : tuple[float, float]
            Wavelengths for fiducial OR readout bit for one tile.
        tile : int or str
            Tile index or tile id.
        round : int or str or None
            Round index or round id.
        bit : int or str or None
            Bit index or bit id.

        Returns
        -------
        wavelengths_um : tuple[float, float] or None
            Wavelengths for fiducial OR readout bit for one tile.
        """
        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if bit is not None:
            if isinstance(bit, int):
                if bit < 0 or bit >= len(self._bit_ids):
                    print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                    return None
                else:
                    local_id = self._bit_ids[bit]
            elif isinstance(bit, str):
                if bit not in self._bit_ids:
                    print("Set valid bit id")
                    return None
                else:
                    local_id = bit
            else:
                print("'bit' must be integer index or string identifier")
                return None
            entity_root = self._readouts_root_path / Path(tile_id) / Path(local_id)
        else:
            if isinstance(round, int):
                if round < 0 or round >= len(self._round_ids):
                    print("Set round index >=0 and <" + str(self._num_rounds))
                    return None
                else:
                    local_id = self._round_ids[round]
            elif isinstance(round, str):
                if round not in self._round_ids:
                    print("Set valid round id")
                    return None
                else:
                    local_id = round
            else:
                print("'round' must be integer index or string identifier")
                return None
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(local_id)

        try:
            if bit is not None:
                from merfish3danalysis.storage.ome import update_chromatic_metadata

                update_chromatic_metadata(
                    self._tile_scene_path,
                    wavelength=(
                        tile_id,
                        local_id,
                        (float(wavelengths_um[0]), float(wavelengths_um[1])),
                    ),
                )
                return None
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "excitation_um": float(wavelengths_um[0]),
                    "emission_um": float(wavelengths_um[1]),
                },
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            print("Error writing wavelength attributes.")
            return None

    def load_local_corrected_image(
        self,
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load gain and offset corrected image for fiducial OR readout bit for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str or None
            Round index or round id.
        bit : int or str or None
            Bit index or bit id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        corrected_image : numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Gain and offset corrected image for fiducial OR readout bit for one tile.
        """
        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if bit is not None:
            if isinstance(bit, int):
                if bit < 0 or bit >= len(self._bit_ids):
                    print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                    return None
                else:
                    local_id = self._bit_ids[bit]
            elif isinstance(bit, str):
                if bit not in self._bit_ids:
                    print("Set valid bit id")
                    return None
                else:
                    local_id = bit
            else:
                print("'bit' must be integer index or string identifier")
                return None
        else:
            if isinstance(round, int):
                if round < 0 or round >= len(self._round_ids):
                    print("Set round index >=0 and <" + str(self._num_rounds))
                    return None
                else:
                    local_id = self._round_ids[round]
            elif isinstance(round, str):
                if round not in self._round_ids:
                    print("Set valid round id")
                    return None
                else:
                    local_id = round
            else:
                print("'round' must be integer index or string identifier")
                return None

        image_path = self.local_image_path(
            tile_id,
            "corrected_data",
            bit=local_id if bit is not None else None,
            round=local_id if round is not None else None,
        )
        if not image_path.exists():
            print("Corrected image not found.")
            return None

        try:
            corrected_image = self._load_from_zarr_array(image_path, return_future)
            return corrected_image
        except (OSError, ValueError):
            print("Error loading corrected image.")
            return None

    def save_local_corrected_image(
        self,
        image: ArrayLike,
        tile: int | str,
        gain_correction: bool = True,
        hotpixel_correction: bool = True,
        shading_correction: bool = False,
        psf_idx: int = 0,
        round: int | str | None = None,
        bit: int | str | None = None,
        return_future: bool | None = False,
        *,
        chunks: Sequence[int] | None = None,
        shards: Sequence[int] | None = None,
    ) -> "WriteFutures | None":
        """Save gain and offset corrected image.

        Parameters
        ----------
        image : ArrayLike
            Local corrected image.
        tile : int or str
            Tile index or tile id.
        gain_correction : bool
            Gain correction applied (True) or not (False).
        hotpixel_correction : bool
            Hotpixel correction applied (True) or not (False).
        shading_correction : bool
            Shading correction applied (True) or not (False).
        psf_idx : int
            PSF index.
        round : int or str or None
            Round index or round id.
        bit : int or str or None
            Bit index or bit id.
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.
        chunks, shards : Sequence[int] or None
            Inner chunk and shard shapes passed to yaozarrs. None selects the
            default chunks and disables sharding, respectively.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None. Invalid tile/round/bit selections also return None.
        """
        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if bit is not None:
            if isinstance(bit, int):
                if bit < 0 or bit >= len(self._bit_ids):
                    print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                    return None
                else:
                    local_id = self._bit_ids[bit]
            elif isinstance(bit, str):
                if bit not in self._bit_ids:
                    print("Set valid bit id")
                    return None
                else:
                    local_id = bit
            else:
                print("'bit' must be integer index or string identifier")
                return None
            entity_root = self._readouts_root_path / Path(tile_id) / Path(local_id)
            round_id = self._round_ids[
                self.load_local_round_linker(tile_id, local_id) - 1
            ]
            frame = f"{tile_id}/{round_id}/{local_id}/native"
        else:
            if isinstance(round, int):
                if round < 0 or round >= len(self._round_ids):
                    print("Set round index >=0 and <" + str(self._num_rounds))
                    return None
                else:
                    local_id = self._round_ids[round]
            elif isinstance(round, str):
                if round not in self._round_ids:
                    print("Set valid round id")
                    return None
                else:
                    local_id = round
            else:
                print("'round' must be integer index or string identifier")
                return None
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(local_id)
            frame = (
                f"{tile_id}/reference"
                if local_id == self._round_ids[0]
                else f"{tile_id}/{local_id}/moving"
            )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root, image_name="corrected_data", image=image
            )
            attributes = self._load_entity_attributes(entity_root)
            attributes.update(
                {
                    "gain_correction": bool(gain_correction),
                    "hotpixel_correction": bool(hotpixel_correction),
                    "shading_correction": bool(shading_correction),
                    "psf_idx": int(psf_idx),
                }
            )
            selection = f"{tile_id}/{local_id}/corrected_data"
            write_future = self._save_to_zarr_array(
                image,
                entity_root / "corrected_data",
                return_future,
                chunks=chunks,
                shards=shards,
                ome_scale=self.voxel_size_zyx_um,
                extra_attributes=attributes,
                scene_path=self._tile_scene_path,
                scene_selection=selection,
                scene_frame=frame,
            )
            return write_future
        except (OSError, TimeoutError, ValueError) as e:
            print(e)
            print("Error saving corrected image.")
            raise

    def load_local_round_transform_zyx_um(
        self,
        tile: int | str,
        round: int | str,
    ) -> ArrayLike | None:
        """
        Load the local fiducial round transform for one tile.

        Parameters
        ----------
        tile : int | str
            Tile index or tile identifier.
        round : int | str
            Fiducial round index or round identifier.

        Returns
        -------
        ArrayLike or None
            Homogeneous 4x4 affine transform in physical Z, Y, X microns. The
            transform maps first-round reference coordinates to coordinates in
            the requested moving round. Returns None when the transform is not
            present.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None
        group = open_group(self._tile_scene_path)
        name = f"{tile_id}/{round_id}/local"
        if group.attrs["qi2lab"]["transform_status"][name] != "calculated":
            return None
        transform = next(
            t
            for t in group.ome_metadata(strict=True).scene.coordinateTransformations
            if t.name == name
        )
        if isinstance(transform, v06.IdentityTransformation):
            return np.eye(4, dtype=np.float32)
        if not isinstance(transform, v06.AffineTransformation):
            raise ValueError("Local registration has an unexpected NGFF type.")
        forward = np.eye(4, dtype=np.float32)
        forward[:3] = np.asarray(transform.affine, dtype=np.float32)
        return np.linalg.inv(forward)

    def save_local_round_transform_zyx_um(
        self,
        transform_zyx_um: ArrayLike,
        tile: int | str,
        round: int | str,
    ) -> None:
        """
        Save the local fiducial round transform for one tile.

        Parameters
        ----------
        transform_zyx_um : ArrayLike
            Homogeneous 4x4 affine transform in physical Z, Y, X microns. The
            transform maps first-round reference coordinates to coordinates in
            the requested moving round.
        tile : int | str
            Tile index or tile identifier.
        round : int | str
            Fiducial round index or round identifier.

        Returns
        -------
        None
            The moving-to-reference affine is stored in the Scene.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None
        from merfish3danalysis.storage.ome import publish_transform

        matrix = np.asarray(transform_zyx_um, dtype=np.float32)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
        ):
            raise ValueError("Local registration must be an invertible 4x4 affine.")
        group = open_group(self._tile_scene_path)
        name = f"{tile_id}/{round_id}/local"
        previous = next(
            t
            for t in group.ome_metadata(strict=True).scene.coordinateTransformations
            if t.name == name
        )
        replacement = v06.AffineTransformation(
            name=name,
            input=previous.input,
            output=previous.output,
            affine=np.linalg.inv(matrix)[:3].tolist(),
        )
        state = group.attrs["qi2lab"]
        publish_transform(
            self._tile_scene_path,
            previous,
            replacement,
            expected_status=state["transform_status"][name],
            expected_revision=state["transform_versions"][name],
        )

    def load_local_sofima_flow_field(
        self,
        *,
        tile: int | str,
        round: int | str,
        return_future: bool | None = True,
    ) -> "tuple[np.ndarray | Future | TensorStore, dict] | None":
        """
        Load the SOFIMA flow field for one local fiducial round.

        Parameters
        ----------
        tile : int or str
            Tile index or tile identifier.
        round : int or str
            Moving fiducial round index or identifier.
        return_future : bool or None, default=True
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        tuple[numpy.ndarray or tensorstore.Future or tensorstore.TensorStore, dict] or None
            SOFIMA flow field and metadata attributes. The map channels are X,
            Y, Z and spatial axes are Z, Y, X. ``map_stride_zyx_px`` is stored
            in Z, Y, X order. ``map_box_start_xyz_px`` is stored in X, Y, Z
            order and gives the reference-grid coordinate of the first flow
            sample. For fields produced by SOFIMA this is the patch center
            coordinate, not the image corner.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        from merfish3danalysis.storage.ome import load_sofima_field

        try:
            return load_sofima_field(
                self._tile_scene_path,
                f"{tile_id}/{round_id}/deformable",
                return_future=return_future,
            )
        except (OSError, ValueError, KeyError) as e:
            print(e)
            print("Error loading SOFIMA flow field.")
            return None

    def save_local_sofima_flow_field(
        self,
        sofima_flow_field_xyz_px: ArrayLike,
        *,
        tile: int | str,
        round: int | str,
        reference_round: int | str,
        map_stride_zyx_px: Sequence[float],
        map_box_start_xyz_px: Sequence[float],
        map_box_size_xyz_px: Sequence[float],
        reference_shape_zyx_px: Sequence[int],
        moving_shape_zyx_px: Sequence[int],
        sofima_status: str = "ok",
        valid_flow_vectors: int | None = None,
        return_future: bool | None = False,
    ) -> "WriteFutures | None":
        """
        Save the SOFIMA flow field for one local fiducial round.

        The saved OME-Zarr image stores the raw float32 SOFIMA map exactly as
        used in memory. The package convention is channel-first ``(3, z, y,
        x)`` with channels ``X, Y, Z`` and spatial axes ``Z, Y, X``. Flow
        values are relative displacements in reference-image pixels from a
        reference coordinate to the affine-initialized moving coordinate.
        ``map_box_start_xyz_px`` is the reference coordinate of the first flow
        sample in ``X, Y, Z`` order. SOFIMA patch-correlation vectors are
        patch-centered, so this is normally half the patch size.

        Parameters
        ----------
        sofima_flow_field_xyz_px : ArrayLike
            Relative SOFIMA flow field with channels X, Y, Z and spatial axes
            Z, Y, X.
        tile : int or str
            Tile index or tile identifier.
        round : int or str
            Moving fiducial round index or identifier.
        reference_round : int or str
            Reference fiducial round index or identifier.
        map_stride_zyx_px : Sequence[float]
            Flow-field stride in reference pixels in Z, Y, X order.
        map_box_start_xyz_px : Sequence[float]
            Reference pixel coordinate of the first flow sample in X, Y, Z
            order.
        map_box_size_xyz_px : Sequence[float]
            Flow-field sample-lattice extent in X, Y, Z order, measured from
            ``map_box_start_xyz_px`` through the last stored map sample.
        reference_shape_zyx_px : Sequence[int]
            Reference image shape in Z, Y, X order.
        moving_shape_zyx_px : Sequence[int]
            Moving native image shape in Z, Y, X order.
        sofima_status : str, default="ok"
            Status reported by the SOFIMA estimator.
        valid_flow_vectors : int or None, default=None
            Number of valid local vectors before missing-vector fill.
        return_future : bool or None, default=False
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None. Invalid tile or round selections also return None.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        if isinstance(reference_round, int):
            if reference_round < 0 or reference_round >= len(self._round_ids):
                print("Set reference round index >=0 and <" + str(len(self._round_ids)))
                return None
            reference_round_id = self._round_ids[reference_round]
        elif isinstance(reference_round, str):
            if reference_round not in self._round_ids:
                print("Set valid reference round id")
                return None
            reference_round_id = reference_round
        else:
            print("'reference_round' must be integer index or string identifier")
            return None
        if reference_round_id != self._round_ids[0]:
            print("SOFIMA reference round must be the first round")
            return None

        image_name = "local_sofima_flow_field"
        attributes = {
            "registration_backend": "sofima",
            "initial_registration_source": "stored_local_affine_transform",
            "flow_field_name": image_name,
            "flow_direction": (
                f"{reference_round_id}_reference_xyz_px_to_affine_initialized_"
                f"{round_id}_xyz_px"
            ),
            "final_render_direction": (
                f"{reference_round_id}_reference_xyz_px_to_moving_native_xyz_px"
            ),
            "flow_representation": "sofima_relative_coordinate_map",
            "flow_channel_order": "xyz",
            "flow_spatial_order": "zyx",
            "map_box_size_xyz_px": np.asarray(
                map_box_size_xyz_px, dtype=np.float32
            ).tolist(),
            "reference_shape_zyx_px": np.asarray(
                reference_shape_zyx_px, dtype=np.int64
            ).tolist(),
            "moving_shape_zyx_px": np.asarray(
                moving_shape_zyx_px, dtype=np.int64
            ).tolist(),
            "sofima_status": str(sofima_status),
            "interpolation_count_final_image": 1,
        }
        if valid_flow_vectors is not None:
            attributes["valid_flow_vectors"] = int(valid_flow_vectors)

        try:
            from merfish3danalysis.storage.ome import publish_sofima_field

            group = open_group(self._tile_scene_path)
            name = f"{tile_id}/{round_id}/deformable"
            previous = next(
                t
                for t in group.ome_metadata(strict=True).scene.coordinateTransformations
                if t.name == name
            )
            state = group.attrs["qi2lab"]
            return publish_sofima_field(
                self._tile_scene_path,
                previous,
                sofima_flow_field_xyz_px,
                spacing_zyx_um=self.voxel_size_zyx_um,
                stride_zyx_px=map_stride_zyx_px,
                start_xyz_px=map_box_start_xyz_px,
                expected_status=state["transform_status"][name],
                expected_revision=state["transform_versions"][name],
                extra_attributes=attributes,
                return_write_future=bool(return_future),
            )
        except (OSError, TimeoutError):
            print("Error saving SOFIMA flow field.")
            raise

    def load_local_deconvolved_fiducial_image(
        self,
        tile: int | str,
        round: int | str,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load a native-frame deconvolved fiducial image.

        Deconvolved fiducials are loaded from ``decon_data`` in their native,
        unwarped tile frame. This image exists only when fiducial deconvolution
        was run during preprocessing.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Native-frame deconvolved fiducial image.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        image_path = self.local_image_path(tile_id, "decon_data", round=round_id)
        if not image_path.exists():
            return None

        try:
            return self._load_from_zarr_array(image_path, return_future)
        except (OSError, ValueError) as e:
            print(e)
            print("Error loading local deconvolved fiducial image.")
            return None

    def load_local_fiducial_image(
        self,
        tile: int | str,
        round: int | str,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load the best available native-frame fiducial image.

        Deconvolved fiducial data are returned when ``decon_data`` exists;
        otherwise corrected fiducial data are returned.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Deconvolved fiducial image if available, otherwise corrected image.
        """
        image = self.load_local_deconvolved_fiducial_image(
            tile=tile,
            round=round,
            return_future=return_future,
        )
        if image is not None:
            return image
        return self.load_local_corrected_image(
            tile=tile,
            round=round,
            return_future=return_future,
        )

    def save_local_deconvolved_fiducial_image(
        self,
        image: ArrayLike,
        tile: int | str,
        round: int | str,
        return_future: bool | None = False,
        *,
        chunks: Sequence[int] | None = None,
        shards: Sequence[int] | None = None,
    ) -> "WriteFutures | None":
        """
        Save a native-frame deconvolved fiducial image.

        Deconvolved fiducials are saved under ``decon_data`` in their native,
        unwarped tile frame.

        Parameters
        ----------
        image : ArrayLike
            Image to save.
        tile : int or str
            Tile index or tile id.
        round : int or str
            Round index or round id.
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.
        chunks, shards : Sequence[int] or None
            Inner chunk and shard shapes passed to yaozarrs. None selects the
            default chunks and disables sharding, respectively.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None. Invalid tile/round/bit selections also return None.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(round, int):
            if round < 0 or round >= len(self._round_ids):
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        entity_root = self._fiducial_root_path / tile_id / round_id
        decon_path = entity_root / Path("decon_data")
        frame = (
            f"{tile_id}/reference"
            if round_id == self._round_ids[0]
            else f"{tile_id}/{round_id}/moving"
        )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name="decon_data",
                image=image,
            )
            attributes = self._load_entity_attributes(entity_root)
            attributes["deconvolution"] = True
            write_future = self._save_to_zarr_array(
                image,
                decon_path,
                return_future,
                chunks=chunks,
                shards=shards,
                ome_scale=self.voxel_size_zyx_um,
                extra_attributes=attributes,
                scene_path=self._tile_scene_path,
                scene_selection=f"{tile_id}/{round_id}/decon_data",
                scene_frame=frame,
            )
            if write_future is not None:
                write_future.result()
            return write_future
        except (OSError, TimeoutError, ValueError):
            print("Error saving local deconvolved fiducial image.")
            raise

    def load_local_deconvolved_readout_image(
        self,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load a native-frame deconvolved readout bit image.

        Deconvolved readout bits are loaded from ``decon_data`` in their
        native, unwarped tile frame. This image exists only when readout
        deconvolution was run during preprocessing.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Native-frame deconvolved readout image.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        image_path = self.local_image_path(tile_id, "decon_data", bit=bit_id)
        if not image_path.exists():
            return None

        try:
            return self._load_from_zarr_array(image_path, return_future)
        except (OSError, ValueError) as e:
            print(e)
            print("Error loading local deconvolved readout image.")
            return None

    def load_local_readout_image(
        self,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load the best available native-frame readout bit image.

        Deconvolved readout data are returned when ``decon_data`` exists;
        otherwise corrected readout data are returned.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Deconvolved readout image if available, otherwise corrected image.
        """
        image = self.load_local_deconvolved_readout_image(
            tile=tile,
            bit=bit,
            return_future=return_future,
        )
        if image is not None:
            return image
        return self.load_local_corrected_image(
            tile=tile,
            bit=bit,
            return_future=return_future,
        )

    def save_local_deconvolved_readout_image(
        self,
        image: ArrayLike,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = False,
        *,
        chunks: Sequence[int] | None = None,
        shards: Sequence[int] | None = None,
    ) -> "WriteFutures | None":
        """
        Save a native-frame deconvolved readout bit image.

        Deconvolved readout bits are saved under ``decon_data`` in their
        native, unwarped tile frame.

        Parameters
        ----------
        image : ArrayLike
            Image to save.
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.
        chunks, shards : Sequence[int] or None
            Inner chunk and shard shapes passed to yaozarrs. None selects the
            default chunks and disables sharding, respectively.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None. Invalid tile/round/bit selections also return None.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        entity_root = self._readouts_root_path / Path(tile_id) / Path(bit_id)
        readout_path = entity_root / Path("decon_data")
        round_id = self._round_ids[self.load_local_round_linker(tile_id, bit_id) - 1]

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name="decon_data",
                image=image,
            )
            attributes = self._load_entity_attributes(entity_root)
            attributes["deconvolution"] = True
            write_future = self._save_to_zarr_array(
                image,
                readout_path,
                return_future,
                chunks=chunks,
                shards=shards,
                ome_scale=self.voxel_size_zyx_um,
                extra_attributes=attributes,
                scene_path=self._tile_scene_path,
                scene_selection=f"{tile_id}/{bit_id}/decon_data",
                scene_frame=f"{tile_id}/{round_id}/{bit_id}/native",
            )
            if write_future is not None:
                write_future.result()
            return write_future
        except (OSError, TimeoutError, ValueError):
            print("Error saving local deconvolved readout image.")
            raise

    def load_local_feature_predictor_image(
        self,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load readout bit feature_predictor prediction image for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        feature_predictor_image : numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            feature_predictor prediction image for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            else:
                bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        image_path = self.local_image_path(
            tile_id, f"{self.feature_predictor_folder_name}_data", bit=bit_id
        )
        if not image_path.exists():
            print("feature_predictor prediction image not found.")
            return None

        try:
            feature_predictor_image = self._load_from_zarr_array(
                image_path, return_future
            )
            return feature_predictor_image
        except (OSError, ValueError) as e:
            print(e)
            print("Error loading feature_predictor image.")
            return None

    def save_local_feature_predictor_image(
        self,
        feature_predictor_image: ArrayLike,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = False,
        *,
        chunks: Sequence[int] | None = None,
        shards: Sequence[int] | None = None,
    ) -> "WriteFutures | None":
        """Save feature_predictor prediction image.

        Parameters
        ----------
        feature_predictor_image : ArrayLike
            feature_predictor prediction image.
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.
        chunks, shards : Sequence[int] or None
            Inner chunk and shard shapes passed to yaozarrs. None selects the
            default chunks and disables sharding, respectively.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None. Invalid tile/round/bit selections also return None.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                local_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            else:
                local_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None
        entity_root = self._readouts_root_path / Path(tile_id) / Path(local_id)
        current_local_zarr_path = entity_root / Path(
            f"{self.feature_predictor_folder_name}_data"
        )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name=f"{self.feature_predictor_folder_name}_data",
                image=feature_predictor_image,
            )
            round_id = self._round_ids[
                self.load_local_round_linker(tile_id, local_id) - 1
            ]
            attributes = self._load_entity_attributes(entity_root)
            write_future = self._save_to_zarr_array(
                feature_predictor_image,
                current_local_zarr_path,
                return_future,
                chunks=chunks,
                shards=shards,
                ome_scale=self.voxel_size_zyx_um,
                extra_attributes=attributes,
                scene_path=self._tile_scene_path,
                scene_selection=f"{tile_id}/{local_id}/{self.feature_predictor_folder_name}_data",
                scene_frame=f"{tile_id}/{round_id}/{local_id}/native",
            )
            if write_future is not None:
                write_future.result()
            return write_future
        except (OSError, ValueError) as e:
            print(e)
            print("Error saving feature_predictor image.")
            raise

    def load_local_feature_predictor_spots(
        self,
        tile: int | str,
        bit: int | str,
    ) -> pd.DataFrame | None:
        """Load feature_predictor spot localizations and features for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.

        Returns
        -------
        feature_predictor_localizations : pandas.DataFrame or None
            feature_predictor localizations and features for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            else:
                bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        current_feature_predictor_localizations_path = (
            self._feature_predictor_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )

        if not current_feature_predictor_localizations_path.exists():
            print("feature_predictor localizations not found.")
            return None
        else:
            feature_predictor_localizations = self._load_from_parquet(
                current_feature_predictor_localizations_path
            )
            return feature_predictor_localizations

    def save_local_feature_predictor_spots(
        self,
        spot_df: pd.DataFrame,
        tile: int | str,
        bit: int | str,
    ) -> None:
        """Save feature_predictor localizations and features.

        Parameters
        ----------
        spot_df : pd.DataFrame
            feature_predictor localizations and features.
        tile : int or str
            Tile index or tile id.
        bit : int or str
            Bit index or bit id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        if isinstance(bit, int):
            if bit < 0 or bit >= len(self._bit_ids):
                print("Set bit index >=0 and <" + str(len(self._bit_ids)))
                return None
            else:
                bit_id = self._bit_ids[bit]
        elif isinstance(bit, str):
            if bit not in self._bit_ids:
                print("Set valid bit id")
                return None
            else:
                bit_id = bit
        else:
            print("'bit' must be integer index or string identifier")
            return None

        tile_localizations_path = (
            self._feature_predictor_localizations_root_path / Path(tile_id)
        )
        tile_localizations_path.mkdir(parents=True, exist_ok=True)

        current_feature_predictor_localizations_path = tile_localizations_path / Path(
            bit_id + ".parquet"
        )

        try:
            self._save_to_parquet(spot_df, current_feature_predictor_localizations_path)
        except OSError as e:
            print(e)
            print("Error saving feature_predictor localizations.")
            return None

    def load_global_coord_xforms_um(
        self,
        tile: int | str,
    ) -> tuple[ArrayLike | None, ArrayLike | None, ArrayLike | None] | None:
        """Load global registration transform for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.

        Returns
        -------
        affine_zyx_um : ArrayLike or None
            Global affine registration transform for one tile.
        origin_zyx_um : ArrayLike or None
            Global origin registration transform for one tile.
        spacing_zyx_um : ArrayLike or None
            Global spacing registration transform for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None, None, None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None, None, None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        group = open_group(self._tile_scene_path)
        names = [
            f"{tile_id}/global",
            f"{tile_id}/decoder_origin",
            f"{tile_id}/decoder_spacing",
        ]
        statuses = group.attrs["qi2lab"]["transform_status"]
        if any(statuses.get(name) != "calculated" for name in names):
            return None, None, None
        transforms = {
            transform.name: transform
            for transform in group.ome_metadata(
                strict=True
            ).scene.coordinateTransformations
        }
        affine, origin, spacing = (transforms[name] for name in names)
        if not (
            isinstance(affine, v06.AffineTransformation)
            and isinstance(origin, v06.TranslationTransformation)
            and isinstance(spacing, v06.ScaleTransformation)
        ):
            raise ValueError("Global coordinate transforms have unexpected NGFF types.")
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3] = np.asarray(affine.affine, dtype=np.float32)
        return (
            matrix,
            np.asarray(origin.translation, dtype=np.float32),
            round_spacing_um(spacing.scale),
        )

    def save_global_coord_xforms_um(
        self,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        tile: int | str,
    ) -> None:
        """Save global registration transform for one tile.

        Parameters
        ----------
        affine_zyx_um : ArrayLike
            Final 4x4 registration correction in physical Z, Y, X coordinates,
            applied after the separately stored camera-to-stage transform.
        origin_zyx_um : ArrayLike
            Native image origin in Z, Y, X micrometers, before camera mapping.
        spacing_zyx_um : ArrayLike
            Native Z, Y, X voxel spacing in micrometers.
        tile : int or str
            Tile index or tile id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        from merfish3danalysis.storage.ome import publish_transforms

        matrix = np.asarray(affine_zyx_um, dtype=np.float32)
        origin = np.asarray(origin_zyx_um, dtype=np.float32)
        spacing = round_spacing_um(spacing_zyx_um)
        if (
            matrix.shape != (4, 4)
            or origin.shape != (3,)
            or spacing.shape != (3,)
            or not np.isfinite(matrix).all()
            or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
        ):
            raise ValueError("Global affine, origin, and spacing must be valid.")
        group = open_group(self._tile_scene_path)
        scene = group.ome_metadata(strict=True)
        names = [
            f"{tile_id}/global",
            f"{tile_id}/decoder_origin",
            f"{tile_id}/decoder_spacing",
        ]
        transforms = {t.name: t for t in scene.scene.coordinateTransformations}
        expected = [transforms[name] for name in names]
        replacements = [
            v06.AffineTransformation(
                name=names[0],
                input=expected[0].input,
                output=expected[0].output,
                affine=matrix[:3].tolist(),
            ),
            v06.TranslationTransformation(
                name=names[1],
                input=expected[1].input,
                output=expected[1].output,
                translation=origin.tolist(),
            ),
            v06.ScaleTransformation(
                name=names[2],
                input=expected[2].input,
                output=expected[2].output,
                scale=spacing.tolist(),
            ),
        ]
        state = group.attrs["qi2lab"]
        publish_transforms(
            self._tile_scene_path,
            expected,
            replacements,
            expected_statuses={name: state["transform_status"][name] for name in names},
            expected_revisions={
                name: state["transform_versions"][name] for name in names
            },
        )

    def save_global_fiducial_projection(
        self,
        projection: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        return_future: bool | None = False,
    ) -> "WriteFutures | None":
        """Publish the fused YX projection in its own NGFF 0.6 Scene.

        The projection has no measured Z. Its world link uses the existing ROI
        convention of pixel Z=10; the original 3D grid remains available for
        the library's ROI calculations.

        Parameters
        ----------
        projection : ArrayLike
            Two-dimensional YX projection pixels.
        affine_zyx_um : ArrayLike
            Homogeneous 4-by-4 affine for the original fused grid.
        origin_zyx_um : ArrayLike
            Original fused-grid origin in ZYX micrometers.
        spacing_zyx_um : ArrayLike
            Original fused-grid spacing in ZYX micrometers.
        return_future : bool or None
            Return the completed TensorStore pixel-write handle when True.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed write handle when requested; otherwise None.
        """
        import os
        from tempfile import TemporaryDirectory

        from merfish3danalysis.storage._scene_update import _scene_lock
        from merfish3danalysis.storage.ome import initialize_scene, publish_image

        pixels = np.asarray(projection)
        affine = np.asarray(affine_zyx_um, dtype=np.float32)
        origin = np.asarray(origin_zyx_um, dtype=np.float32)
        spacing = round_spacing_um(spacing_zyx_um)
        if (
            pixels.ndim != 2
            or affine.shape != (4, 4)
            or origin.shape != (3,)
            or spacing.shape != (3,)
        ):
            raise ValueError(
                "Projection must be YX with a 3D affine, origin and spacing."
            )
        if (
            not np.isfinite(affine).all()
            or not np.isfinite(origin).all()
            or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)
        ):
            raise ValueError(
                "Projection coordinates must be finite with positive spacing."
            )
        image = v06.Image(
            multiscales=[
                v06.Multiscale.from_dims(
                    [
                        DimSpec(
                            name=axis,
                            size=size,
                            unit="micrometer",
                            scale=float(step),
                            translation=float(start),
                        )
                        for axis, size, step, start in zip(
                            "yx", pixels.shape, spacing[1:], origin[1:], strict=True
                        )
                    ]
                )
            ]
        )
        attrs = {
            "affine_zyx_um": affine.tolist(),
            "origin_zyx_um": origin.tolist(),
            "spacing_zyx_um": spacing.tolist(),
        }
        path = self.fiducial_projection_path()

        def publish(dest: Path, expected_revision: str | None) -> Any:
            return publish_image(
                dest,
                "projection/fiducial",
                "world",
                image,
                pixels,
                expected_revision=expected_revision,
                extra_attributes=attrs,
                return_write_future=bool(return_future),
                affine_zyx_um=affine,
                missing_z_origin_um=float(origin[0] + 10 * spacing[0]),
            )

        if not path.exists():
            with TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temp:
                candidate = Path(temp) / path.name
                initialize_scene(candidate)
                published = publish(candidate, None)
                open_group(candidate).validate(strict=True)
                with _scene_lock(path.resolve()):
                    if path.exists():
                        raise FileExistsError(
                            "Projection Scene was created by another worker."
                        )
                    os.rename(candidate, path)
        else:
            products = open_group(path).attrs["qi2lab"]["products"]
            previous = products.get("projection/fiducial")
            published = publish(
                path, None if previous is None else previous["revision"]
            )
        return published[1] if return_future else None

    def load_global_fiducial_projection(
        self, return_future: bool | None = True
    ) -> "tuple[np.ndarray | Future | TensorStore, np.ndarray, np.ndarray, np.ndarray] | None":
        """Read the projection and its original fused 3D ROI coordinate metadata.

        Parameters
        ----------
        return_future : bool or None
            True returns a TensorStore read future, False a NumPy array, and
            None a sliceable TensorStore handle.

        Returns
        -------
        tuple or None
            Projection, affine, ZYX origin, and ZYX spacing; None if missing.
        """
        path = self.fiducial_projection_path()
        if not path.exists():
            return None
        scene = open_group(path)
        product = scene.attrs["qi2lab"]["products"].get("projection/fiducial")
        if product is None:
            return None
        image_path = path / product["path"]
        attrs = self.load_image_metadata(image_path)
        return (
            self._load_from_zarr_array(image_path, return_future),
            np.asarray(attrs["affine_zyx_um"], dtype=np.float32),
            np.asarray(attrs["origin_zyx_um"], dtype=np.float32),
            round_spacing_um(attrs["spacing_zyx_um"]),
        )

    def load_global_fused_image(
        self,
        fusion_type: str = "fiducial",
        return_future: bool | None = True,
    ) -> "tuple[np.ndarray | Future | TensorStore, dict[str, Any]] | None":
        """Load a library-produced fused image and its metadata through the datastore.

        ``return_future=None`` provides a lazy TensorStore for consumers such as
        the fused viewer and TIFF export. ``all_channels`` selects the fusion
        workflow output or the product written by the existing image setter.

        Parameters
        ----------
        fusion_type : str
            ``"fiducial"`` or ``"all_channels"``.
        return_future : bool or None
            True returns a TensorStore read future, False a NumPy array, and
            None a sliceable TensorStore handle.

        Returns
        -------
        tuple or None
            Fused image and image metadata, or None when unavailable.
        """
        if fusion_type == "fiducial":
            paths = [self.fused_image_path()]
        elif fusion_type == "all_channels":
            paths = [self.fused_image_path("full_dataset")]
        else:
            raise ValueError("fusion_type must be 'fiducial' or 'all_channels'.")
        image_path = next((path for path in paths if path.exists()), None)
        if image_path is None:
            return None
        return (
            self._load_from_zarr_array(image_path, return_future),
            self.load_image_metadata(image_path),
        )

    def load_global_fiducial_image(
        self,
        return_future: bool | None = True,
    ) -> "tuple[np.ndarray | Future | TensorStore, ArrayLike, ArrayLike, ArrayLike] | None":
        """Load downsampled, fused fiducial image.

        Parameters
        ----------
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        fused_image : numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Downsampled, fused fiducial image.
        affine_zyx_um : ArrayLike or None
            Global affine registration transform for fused image.
        origin_zyx_um : ArrayLike or None
            Global origin registration transform for fused image.
        spacing_zyx_um : ArrayLike or None
            Global spacing registration transform for fused image.
        """
        try:
            product = self.load_global_fused_image(return_future=return_future)
            if product is None:
                print("Globally registered, fused image not found.")
                return None
            fused_image, attributes = product
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = round_spacing_um(attributes["spacing_zyx_um"])
            return fused_image, affine_zyx_um, origin_zyx_um, spacing_zyx_um
        except (OSError, ValueError, KeyError):
            print("Error loading globally registered, fused image.")
            return None

    def save_global_fiducial_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        fusion_type: str = "fiducial",
        return_future: bool | None = False,
    ) -> "WriteFutures | None":
        """Save downsampled, fused fiducial image.

        The caller selects the fusion grid. This method stores the supplied
        image and spacing without choosing a dataset-specific downsampling.

        Parameters
        ----------
        fused_image : ArrayLike
            Downsampled, fused fiducial image.
        affine_zyx_um : ArrayLike
            Global affine registration transform for fused image.
        origin_zyx_um : ArrayLike
            Global origin registration transform for fused image.
        spacing_zyx_um : ArrayLike
            Caller-selected fused voxel spacing in Z, Y, X microns.
        fusion_type : str
            Type of fusion (fiducial or all_channels).
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None.
        """
        scene_path = self._ensure_fused_scene(fusion_type)
        current_local_zarr_path = scene_path / "image"

        metadata_attrs = {
            "affine_zyx_um": np.asarray(affine_zyx_um, dtype=np.float32).tolist(),
            "origin_zyx_um": np.asarray(origin_zyx_um, dtype=np.float32).tolist(),
            "spacing_zyx_um": round_spacing_um(spacing_zyx_um).tolist(),
        }
        fused_array = np.asarray(fused_image)
        try:
            write_future = self._save_to_zarr_array(
                fused_array.astype(np.uint16),
                current_local_zarr_path,
                return_future,
                ome_scale=metadata_attrs["spacing_zyx_um"],
                ome_translation=metadata_attrs["origin_zyx_um"],
                extra_attributes=metadata_attrs,
                chunks=self._fused_image_chunks(fused_array),
                scene_path=scene_path,
                scene_selection=self._FUSED_SCENE_SELECTIONS[fusion_type],
                scene_frame="world",
                scene_affine=affine_zyx_um,
            )
            if write_future is not None:
                write_future.result()
            return write_future
        except (OSError, TimeoutError):
            print("Error saving fused image.")
            raise

    def load_local_decoded_spots(
        self,
        tile: int | str,
        decode_run_key: str | None = None,
    ) -> pd.DataFrame | None:
        """Load decoded spots and features for one tile.

        Parameters
        ----------
        tile : int or str
            Tile index or tile id.
        decode_run_key : str or None, default None
            Optional decoded-output run key.

        Returns
        -------
        tile_features : pandas.DataFrame or None
            Decoded spots and features for one tile.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        current_tile_features_path = self._decoded_run_root(decode_run_key) / Path(
            tile_id + ".parquet"
        )

        if not current_tile_features_path.exists():
            print("Decoded spots not found.")
            return None
        else:
            tile_features = self._load_from_parquet(current_tile_features_path)
            return tile_features

    def save_local_decoded_spots(
        self,
        features_df: pd.DataFrame,
        tile: int | str,
        decode_run_key: str | None = None,
    ) -> None:
        """Save decoded spots and features for one tile.

        Parameters
        ----------
        features_df : pd.DataFrame
            Decoded spots and features for one tile.
        tile : int or str
            Tile index or tile id.
        decode_run_key : str or None, default None
            Optional decoded-output run key.
        """
        if isinstance(tile, int):
            if tile < 0 or tile >= self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tiled id")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        decoded_root_path = self._decoded_run_root(decode_run_key)
        decoded_root_path.mkdir(parents=True, exist_ok=True)
        current_tile_features_path = decoded_root_path / Path(tile_id + ".parquet")

        self._save_to_parquet(features_df, current_tile_features_path)

    def load_global_filtered_decoded_spots(
        self,
        decode_run_key: str | None = None,
        gene_ids: Collection[str] | None = None,
        columns: Collection[str] | None = None,
    ) -> pd.DataFrame | None:
        """Load all decoded and filtered spots.

        Parameters
        ----------
        decode_run_key : str or None, default None
            Optional decoded-output run key.
        gene_ids : Collection[str] or None, default None
            Optional gene ids to load.
        columns : Collection[str] or None, default None
            Optional columns to load.

        Returns
        -------
        all_tiles_filtered : pandas.DataFrame or None
            All decoded and filtered spots.
        """
        current_global_filtered_decoded_dir_path = self._global_filtered_decoded_root(
            decode_run_key
        )
        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("decoded_features.parquet")
        )

        if not current_global_filtered_decoded_path.exists():
            print("Global, filtered, decoded spots not found.")
            return None
        elif gene_ids is not None or columns is not None:
            selected_gene_ids = (
                [str(gene_id) for gene_id in gene_ids] if gene_ids is not None else []
            )
            filters = (
                [("gene_id", "in", selected_gene_ids)] if gene_ids is not None else None
            )
            all_tiles_filtered = pd.read_parquet(
                current_global_filtered_decoded_path,
                columns=None if columns is None else list(columns),
                filters=filters,
            )
            if gene_ids is not None and "gene_id" in all_tiles_filtered.columns:
                all_tiles_filtered = all_tiles_filtered[
                    all_tiles_filtered["gene_id"].astype(str).isin(selected_gene_ids)
                ]
            return all_tiles_filtered
        else:
            all_tiles_filtered = self._load_from_parquet(
                current_global_filtered_decoded_path
            )
            return all_tiles_filtered

    def save_global_filtered_decoded_spots(
        self,
        filtered_decoded_df: pd.DataFrame,
        decode_run_key: str | None = None,
    ) -> None:
        """Save all decoded and filtered spots.

        Parameters
        ----------
        filtered_decoded_df : pd.DataFrame
            All decoded and filtered spots.
        decode_run_key : str or None, default None
            Optional decoded-output run key.
        """
        current_global_filtered_decoded_dir_path = self._global_filtered_decoded_root(
            decode_run_key
        )

        if not current_global_filtered_decoded_dir_path.exists():
            current_global_filtered_decoded_dir_path.mkdir(parents=True)

        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("decoded_features.parquet")
        )
        current_global_filtered_decoded_csv_gz_path = (
            current_global_filtered_decoded_dir_path / Path("decoded_features.csv.gz")
        )

        self._save_to_parquet(filtered_decoded_df, current_global_filtered_decoded_path)
        self._save_to_csv_gz(
            filtered_decoded_df, current_global_filtered_decoded_csv_gz_path
        )

    def load_global_cellpose_outlines(
        self,
    ) -> dict | None:
        """Load Cellpose max projection cell outlines.

        Returns
        -------
        cellpose_outlines : dict or None
            Cellpose cell mask outlines.
        """
        current_cellpose_outlines_path = self.cellpose_outlines_path()

        if not current_cellpose_outlines_path.exists():
            print("Cellpose cell mask outlines not found.")
            return None
        else:
            cellpose_outlines = self._load_from_microjson(
                current_cellpose_outlines_path
            )
            return cellpose_outlines

    def load_global_cellpose_roi_zip(
        self,
    ) -> dict[int, ArrayLike] | None:
        """Load global Cellpose outlines from an ImageJ ROI zip.

        Returns
        -------
        cellpose_outlines : dict[int, ArrayLike] or None
            Cellpose ROI outlines with global X, Y coordinates.
        """
        roi_path = self.cellpose_global_rois_path()
        if not roi_path.exists():
            print("Cellpose global ROI zip not found.")
            return None

        try:
            from roifile import roiread
        except ImportError:
            print("roifile is required to load Cellpose ROI zip.")
            return None

        outlines: dict[int, ArrayLike] = {}
        try:
            for idx, roi in enumerate(roiread(roi_path)):
                coordinates = getattr(roi, "subpixel_coordinates", None)
                if coordinates is None:
                    coordinates = roi.coordinates()
                if coordinates is not None:
                    outlines[idx + 1] = np.asarray(coordinates, dtype=float)
            return outlines
        except (OSError, ValueError) as exc:
            print(exc)
            print("Error loading Cellpose ROI zip.")
            return None

    def _proseg_3d_root(self) -> Path:
        """
        Return the root path for Proseg 3D outputs.

        Returns
        -------
        Path
            Proseg 3D output root.
        """
        return self._datastore_path / Path("proseg") / Path("3D")

    def _proseg_3d_run_root(self, run_name: str | None = None) -> Path:
        """
        Return one Proseg 3D run root path.

        Parameters
        ----------
        run_name : str or None, default None
            Proseg run name. ``None`` or ``"default"`` selects ``proseg/3D``.

        Returns
        -------
        Path
            Proseg run root path.
        """
        proseg_root = self._proseg_3d_root()
        if run_name is None or run_name == "default":
            return proseg_root
        return proseg_root / Path(run_name)

    def list_proseg_3d_runs(self) -> list[str]:
        """List available Proseg 3D output runs.

        Returns
        -------
        proseg_runs : list[str]
            Available Proseg 3D run names. The direct ``proseg/3D`` output is
            named ``"default"``.
        """
        proseg_root = self._proseg_3d_root()
        if not proseg_root.exists():
            return []

        run_names: list[str] = []
        candidate_roots = [proseg_root]
        candidate_roots.extend(
            path for path in sorted(proseg_root.iterdir()) if path.is_dir()
        )
        for root in candidate_roots:
            transcript_path = root / Path("transcript_metadata_3D.csv.gz")
            polygon_path = root / Path("cell_polygons_3D.geojson.gz")
            if transcript_path.exists() and polygon_path.exists():
                run_names.append("default" if root == proseg_root else root.name)
        return sorted(run_names)

    def load_proseg_transcripts_3d(
        self,
        run_name: str | None = None,
    ) -> pd.DataFrame | None:
        """Load Proseg 3D transcript metadata.

        Parameters
        ----------
        run_name : str or None, default None
            Proseg run name. ``None`` or ``"default"`` selects ``proseg/3D``.

        Returns
        -------
        transcripts : pandas.DataFrame or None
            Proseg transcript metadata.
        """
        transcript_path = self._proseg_3d_run_root(run_name) / Path(
            "transcript_metadata_3D.csv.gz"
        )
        if not transcript_path.exists():
            print("Proseg transcript metadata not found.")
            return None
        try:
            return pd.read_csv(transcript_path, compression="gzip")
        except (OSError, ValueError) as exc:
            print(exc)
            print("Error loading Proseg transcript metadata.")
            return None

    def load_proseg_cell_polygons_3d(
        self,
        run_name: str | None = None,
    ) -> dict[Any, ArrayLike] | None:
        """Load Proseg 3D refined cell polygons.

        Parameters
        ----------
        run_name : str or None, default None
            Proseg run name. ``None`` or ``"default"`` selects ``proseg/3D``.

        Returns
        -------
        polygons : dict[Any, ArrayLike] or None
            Cell identifiers mapped to polygon vertices in X, Y order.
        """
        polygon_path = self._proseg_3d_run_root(run_name) / Path(
            "cell_polygons_3D.geojson.gz"
        )
        if not polygon_path.exists():
            print("Proseg cell polygons not found.")
            return None

        try:
            with gzip.open(polygon_path, "rt", encoding="utf-8") as file:
                geojson = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(exc)
            print("Error loading Proseg cell polygons.")
            return None

        polygons: dict[Any, ArrayLike] = {}
        for idx, feature in enumerate(geojson.get("features", [])):
            properties = feature.get("properties", {})
            cell_id = properties.get("cell", idx)
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            rings: list[np.ndarray] = []
            if geometry.get("type") == "Polygon":
                rings.extend(np.asarray(ring, dtype=float) for ring in coordinates)
            elif geometry.get("type") == "MultiPolygon":
                for polygon in coordinates:
                    rings.extend(np.asarray(ring, dtype=float) for ring in polygon)
            if rings:
                polygons[cell_id] = max(rings, key=len)

        return polygons

    def baysor_3d_available(self) -> bool:
        """Return whether Baysor 3D molecules and cell boundaries are present."""
        baysor_root = self._datastore_path / "baysor/3D"
        return (baysor_root / Path("molecules.parquet")).exists() and (
            baysor_root / Path("cell_boundaries_3d.parquet")
        ).exists()

    def load_baysor_molecules_3d(self) -> pd.DataFrame | None:
        """
        Load Baysor 3D molecule assignments.

        Returns
        -------
        pandas.DataFrame or None
            Baysor molecule table, if present.
        """
        molecules_path = (
            self._datastore_path
            / Path("baysor")
            / Path("3D")
            / Path("molecules.parquet")
        )
        if not molecules_path.exists():
            print("Baysor molecules not found.")
            return None
        try:
            return self._load_from_parquet(molecules_path)
        except (OSError, ValueError) as exc:
            print(exc)
            print("Error loading Baysor molecules.")
            return None

    def load_baysor_cell_boundaries_3d(self) -> pd.DataFrame | None:
        """
        Load Baysor 3D cell boundary polygons.

        Returns
        -------
        pandas.DataFrame or None
            Baysor cell boundary table, if present.
        """
        boundaries_path = (
            self._datastore_path
            / Path("baysor")
            / Path("3D")
            / Path("cell_boundaries_3d.parquet")
        )
        if not boundaries_path.exists():
            print("Baysor cell boundaries not found.")
            return None
        try:
            return self._load_from_parquet(boundaries_path)
        except (OSError, ValueError) as exc:
            print(exc)
            print("Error loading Baysor cell boundaries.")
            return None

    def load_global_cellpose_segmentation_image(
        self,
        return_future: bool | None = True,
    ) -> "np.ndarray | Future | TensorStore | None":
        """Load Cellpose max projection, downsampled segmentation image.

        Parameters
        ----------
        return_future : bool or None
            True returns a read future, False returns a NumPy array, and None
            returns a sliceable TensorStore handle opened through yaozarrs.

        Returns
        -------
        fused_image : numpy.ndarray or tensorstore.Future or tensorstore.TensorStore or None
            Cellpose max projection, downsampled segmentation image.
        """
        path = self.fiducial_projection_path()
        if not path.exists():
            return None
        try:
            scene = open_group(path)
            product = scene.attrs["qi2lab"]["products"]["projection/fiducial"]
            labels_path = path / product["path"] / "labels"
            if not labels_path.exists():
                return None
            labels = open_group(labels_path).ome_metadata(strict=True).labels
            name = next((item for item in labels if item.startswith("cellpose_")), None)
            if name is None:
                return None
            return self._load_from_zarr_array(labels_path / name, return_future)
        except (OSError, ValueError, KeyError):
            print("Error loading Cellpose image.")
            return None

    def save_global_cellpose_segmentation_image(
        self,
        cellpose_image: ArrayLike,
        downsampling: Sequence[float],
        return_future: bool | None = False,
    ) -> "WriteFutures | None":
        """Save Cellpose max projection, downsampled segmentation image.

        The caller runs segmentation and selects its image grid. This method
        stores the supplied labels unchanged and derives their physical scale
        from the native voxel calibration and supplied downsampling factors.

        Parameters
        ----------
        cellpose_image : ArrayLike
            Cellpose max projection, downsampled segmentation image.
        downsampling : Sequence[float]
            Caller-selected Z, Y, X factors: segmentation spacing divided by
            native voxel spacing. Two-dimensional masks use the resulting YX
            physical scale while retaining all three factors in metadata.
        return_future : bool or None
            Return the completed TensorStore write handle when True. Writes
            finish before associated metadata is reported as saved.

        Returns
        -------
        tensorstore.WriteFutures or None
            Completed pixel-write handle when return_future is True; otherwise
            None.
        """
        downsampling = np.asarray(downsampling, dtype=np.float64)
        if (
            downsampling.shape != (3,)
            or not np.all(np.isfinite(downsampling))
            or np.any(downsampling <= 0)
        ):
            raise ValueError(
                "Downsampling must contain three positive finite Z, Y, X factors."
            )
        projection = self.load_global_fiducial_projection(return_future=None)
        if projection is None:
            raise ValueError(
                "A fused fiducial projection is required before saving labels."
            )
        _, _, _, spacing = projection
        if not np.allclose(
            round_spacing_um(self.voxel_size_zyx_um) * downsampling, spacing
        ):
            raise ValueError("Label downsampling must match the projection grid.")
        from merfish3danalysis.storage.ome import publish_projection_labels

        return publish_projection_labels(
            self.fiducial_projection_path(),
            cellpose_image,
            return_write_future=bool(return_future),
        )
