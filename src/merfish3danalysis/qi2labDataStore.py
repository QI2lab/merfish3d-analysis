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

import json
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import TimeoutError
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from roifile import ROI_TYPE, ImagejRoi, roiread, roiwrite
from shapely.geometry import Point, Polygon

try:
    from zarr.errors import ZarrError
except Exception:
    ZarrError = Exception


class qi2labDataStore:
    """API to qi2lab MERFISH store.

    Parameters
    ----------
    datastore_path : Union[str, Path]
        Path to qi2lab MERFISH store

    """

    def __init__(self, datastore_path: str | Path) -> None:
        compressor = {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 2,
        }
        self._zarrv2_spec = {
            "driver": "zarr",
            "kvstore": None,
            "metadata": {"compressor": compressor},
            "open": True,
            "assume_metadata": False,
            "create": True,
            "delete_existing": False,
        }

        self._datastore_path = Path(datastore_path)
        if self._datastore_path.exists():
            self._parse_datastore()
        else:
            self._init_datastore()

    @property
    def datastore_state(self) -> dict | None:
        """Datastore state.

        Returns
        -------
        datastore_state : Optional[dict]
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

        if not hasattr(self, "_datastore_state") or self._datastore_state is None:
            self._datastore_state = value
        else:
            self._datastore_state.update(value)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    def _calibrations_attributes_path(self) -> Path:
        """Path to calibrations metadata sidecar."""

        return self._calibrations_zarr_path / Path("attributes.json")

    def _load_calibrations_attributes(self) -> dict[str, Any]:
        """Load calibrations metadata sidecar."""

        attributes = self._load_from_json(self._calibrations_attributes_path())
        if not isinstance(attributes, dict):
            raise ValueError("calibrations/attributes.json is invalid.")
        return attributes

    def _save_calibrations_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Persist calibrations metadata sidecar."""

        self._save_to_json(
            {str(k): self._to_json_compatible(v) for k, v in dict(attributes).items()},
            self._calibrations_attributes_path(),
        )

    def _set_calibration_attribute(self, key: str, value: Any) -> None:
        """Update one calibration metadata field."""

        attributes = self._load_calibrations_attributes()
        attributes[str(key)] = self._to_json_compatible(value)
        self._save_calibrations_attributes(attributes)

    @staticmethod
    def _strict_id_sort_key(name: str, prefix: str, width: int) -> int:
        """Validate and parse strict zero-padded identifiers."""

        match = re.fullmatch(rf"{re.escape(prefix)}(\d{{{width}}})", name)
        if match is None:
            raise ValueError(
                f"Invalid identifier '{name}'. Expected '{prefix}' followed by {width} digits."
            )
        return int(match.group(1))

    @classmethod
    def _collect_strict_ids(cls, parent: Path, prefix: str, width: int) -> list[str]:
        """Collect and sort strict identifiers under a folder."""

        ids = [entry.name for entry in parent.iterdir() if entry.is_dir()]
        ids.sort(key=lambda value: cls._strict_id_sort_key(value, prefix, width))
        return ids

    @property
    def microscope_type(self) -> str | None:
        """Microscope type.

        Returns
        -------
        microscope_type : Optional[str]
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
        camera_model : Optional[str]
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
        -------
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

        self._num_tiles = value
        self._set_calibration_attribute("num_tiles", value)

        self._tile_ids = []
        for tile_idx in range(value):
            self._tile_ids.append("tile" + str(tile_idx).zfill(4))

    @property
    def channels_in_data(self) -> Collection[int] | None:
        """Channel indices.

        Returns
        -------
        channels_in_data : Collection[int]
            Channel indices.
        """

        return getattr(self, "_channels_in_data", None)

    @channels_in_data.setter
    def channels_in_data(self, value: Collection[int]) -> None:
        """Set the channels in the data.

        Parameters
        ----------
        value : Collection[int]
            New channels in data (int values starting from zero).
        """

        self._channels_in_data = value
        self._set_calibration_attribute("channels_in_data", value)

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
            Electrons per camera ADU."""

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

        return getattr(self, "_noise_map", None)

    @noise_map.setter
    def noise_map(self, value: ArrayLike) -> None:
        """Set the camera noise image.

        Parameters
        ----------
        value : ArrayLike
            New camera noise image.
        """

        self._noise_map = value
        current_local_zarr_path = str(self._calibrations_zarr_path / Path("noise_map"))

        try:
            self._save_to_zarr_array(
                value,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future=False,
            )
        except (OSError, ZarrError):
            print(r"Could not access calibrations/noise_map")

    @property
    def channel_shading_maps(self) -> ArrayLike | None:
        """Channel shaiding images.

        Returns
        -------
        channel_shading_maps : ArrayLike
            Channel shading images.
        """

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
        if shading_maps.shape[0] > 1:
            reference_shape = tuple(shading_maps[0].shape)
            for channel_map in shading_maps[1:]:
                if tuple(channel_map.shape) != reference_shape:
                    raise ValueError(
                        "All shading maps must share the same YX shape "
                        f"(expected {reference_shape}, got {tuple(channel_map.shape)})."
                    )

        self._shading_maps = shading_maps
        current_local_zarr_path = str(
            self._calibrations_zarr_path / Path("shading_maps")
        )

        try:
            self._save_to_zarr_array(
                shading_maps,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future=False,
            )
        except (OSError, ZarrError):
            print(r"Could not access calibrations/shading_maps")

    @property
    def channel_psfs(self) -> ArrayLike | None:
        """Channel point spread functions (PSF).

        Return
        ------
        channel_psfs : ArrayLike
            Channel point spread functions (PSF).
        """

        psfs = getattr(self, "_psfs", None)
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

        self._psfs = psf_list
        psf_root_path = self._calibrations_zarr_path / Path("psf_data")
        psf_root_path.mkdir(exist_ok=True, parents=True)
        psf_manifest: dict[str, Any] = {}

        try:
            for psf_idx, psf_array in enumerate(psf_list):
                psf_id = f"psf_{psf_idx:03d}"
                current_psf_path = psf_root_path / Path(psf_id)
                self._save_to_zarr_array(
                    psf_array,
                    self._get_kvstore_key(current_psf_path),
                    self._zarrv2_spec.copy(),
                    return_future=False,
                )
                psf_manifest[str(psf_idx)] = {
                    "id": psf_id,
                    "shape_zyx": list(psf_array.shape),
                }

            self._set_calibration_attribute("psf_manifest", psf_manifest)
        except (OSError, ValueError):
            print(r"Could not access calibrations/psf_data")

    @property
    def experiment_order(self) -> pd.DataFrame | None:
        """Round and bit order.

        Returns
        -------
        experiment_order : pd.DataFrame
            Round and bit order.
        """

        return getattr(self, "_experiment_order", None)

    @experiment_order.setter
    def experiment_order(self, value: ArrayLike | pd.DataFrame) -> None:
        """Set the round and bit order.

        Parameters
        ----------
        value : Union[ArrayLike, pd.DataFrame]
            New round and bit order.
        """

        if isinstance(value, pd.DataFrame):
            self._experiment_order = value
        else:
            channel_list = []
            for idx in range(len(self._channels_in_data)):
                channel_list.append(str(self._channels_in_data[idx]))
            self._experiment_order = pd.DataFrame(
                value, columns=channel_list, dtype="int64"
            )

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

        return getattr(self, "_voxel_size_zyx_um", None)

    @voxel_size_zyx_um.setter
    def voxel_size_zyx_um(self, value: ArrayLike) -> None:
        """Set the voxel size, zyx order (microns).

        Parameters
        ----------
        value : ArrayLike
            New voxel size, zyx order (microns).
        """

        self._voxel_size_zyx_um = value
        self._set_calibration_attribute("voxel_size_zyx_um", value)

    @property
    def baysor_path(self) -> Path | str:
        """Baysor path

        Returns
        -------
        baysor_path : Union[Path,str]
            Baysor path.
        """

        return getattr(self, "_baysor_path", None)

    @baysor_path.setter
    def baysor_path(self, value: Path | str) -> None:
        """Set the baysor path.

        Parameters
        ----------
        value : Union[Path,str]
            New baysor path.
        """

        if value is None:
            self._baysor_path = None
            self._datastore_state["BaysorPath"] = None
        else:
            self._baysor_path = Path(value)
            self._datastore_state["BaysorPath"] = str(self._baysor_path)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @property
    def baysor_options(self) -> Path | str:
        """Baysor options

        Returns
        -------
        baysor_options : Union[Path,str]
            Baysor options.
        """
        return getattr(self, "_baysor_options", None)

    @baysor_options.setter
    def baysor_options(self, value: Path | str) -> None:
        """Set the baysor options.

        Parameters
        ----------
        value : Union[Path,str]
            New baysor options.
        """

        if value is None:
            self._baysor_path = None
            self._datastore_state["BaysorPath"] = None
        else:
            self._baysor_options = Path(value)
            self._datastore_state["BaysorOptions"] = str(self._baysor_options)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @property
    def julia_threads(self) -> int:
        """Julia thread number

        Returns
        -------
        julia_threads : int
            Julia thread number.
        """

        return getattr(self, "_julia_threads", None)

    @julia_threads.setter
    def julia_threads(self, value: int) -> None:
        """Set the julia thread number.

        Parameters
        ----------
        value : int
            New julia thread number.
        """

        self._julia_threads = value
        self._datastore_state["JuliaThreads"] = str(self._julia_threads)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

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

        self._datastore_path.mkdir(parents=True)
        self._calibrations_zarr_path = self._datastore_path / Path(r"calibrations")
        self._calibrations_zarr_path.mkdir()
        self._save_to_json({}, self._calibrations_attributes_path())
        self.fiducial_folder_name = r"fiducial"
        self._fiducial_root_path = self._datastore_path / Path(
            self.fiducial_folder_name
        )
        self._fiducial_root_path.mkdir()
        self._readouts_root_path = self._datastore_path / Path(r"readouts")
        self._readouts_root_path.mkdir()
        self.feature_predictor_folder_name = r"feature_predictor"
        self._feature_predictor_localizations_root_path = self._datastore_path / Path(
            f"{self.feature_predictor_folder_name}_localizations"
        )
        self._feature_predictor_localizations_root_path.mkdir()
        self._decoded_root_path = self._datastore_path / Path(r"decoded")
        self._decoded_root_path.mkdir()
        self._fused_root_path = self._datastore_path / Path(r"fused")
        self._fused_root_path.mkdir()
        self._segmentation_root_path = self._datastore_path / Path(r"segmentation")
        self._segmentation_root_path.mkdir()
        self._mtx_output_root_path = self._datastore_path / Path(r"mtx_output")
        self._mtx_output_root_path.mkdir()
        self._baysor_path = r""
        self._baysor_options = r""
        self._julia_threads = 0

        # initialize datastore state
        self._datastore_state_json_path = self._datastore_path / Path(
            r"datastore_state.json"
        )
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
            "RefinedSpots": False,
            "mtxOutput": False,
            "BaysorPath": str(self._baysor_path),
            "BaysorOptions": str(self._baysor_options),
            "JuliaThreads": str(self._julia_threads),
        }

        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @staticmethod
    def _get_kvstore_key(path: Path | str) -> dict:
        """Convert datastore location to tensorstore kvstore key.

        Parameters
        ----------
        path : Union[Path, str]
            Datastore location.

        Returns
        -------
        kvstore_key : dict
            Tensorstore kvstore key.
        """

        path_str = str(path)
        if path_str.startswith("s3://") or "s3.amazonaws.com" in path_str:
            return {"driver": "s3", "path": path_str}
        elif path_str.startswith("gs://") or "storage.googleapis.com" in path_str:
            return {"driver": "gcs", "path": path_str}
        elif path_str.startswith("azure://") or "blob.core.windows.net" in path_str:
            return {"driver": "azure", "path": path_str}
        elif path_str.startswith("http://") or path_str.startswith("https://"):
            raise ValueError("Unsupported cloud storage provider in URL")
        else:
            return {"driver": "file", "path": path_str}

    @staticmethod
    def _import_yaozarrs() -> tuple[Any, Any, Any]:
        """Import yaozarrs lazily so module import remains lightweight."""

        try:
            from yaozarrs import open_group, v05
            from yaozarrs.write.v05 import write_image
        except Exception as exc:
            raise ImportError(
                "yaozarrs is required for datastore image IO. "
                "Install yaozarrs with tensorstore write support."
            ) from exc
        return open_group, v05, write_image

    @staticmethod
    def _extract_local_path_from_kvstore(kvstore: dict | Path | str) -> Path:
        """Extract a local filesystem path from a kvstore-like input."""

        if isinstance(kvstore, (str, Path)):
            return Path(kvstore)
        if isinstance(kvstore, dict):
            if kvstore.get("driver") == "file":
                return Path(str(kvstore["path"]))
            raise ValueError(
                "Only local file kvstores are supported for datastore image IO."
            )
        raise TypeError(f"Unsupported kvstore type: {type(kvstore)!r}")

    @staticmethod
    def _create_array_tensorstore_qi2lab(
        path: Path,
        shape: tuple[int, ...],
        dtype: Any,
        chunks: tuple[int, ...],
        *,
        shards: tuple[int, ...] | None,
        dimension_names: list[str] | None,
        overwrite: bool,
        compression: str,
    ) -> Any:
        """Create zarr3 arrays with qi2lab compression defaults via tensorstore."""

        import tensorstore as ts

        if compression == "blosc-zstd":
            chunk_codecs = [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": 5,
                        "shuffle": "bitshuffle",
                    },
                }
            ]
        elif compression == "blosc-lz4":
            chunk_codecs = [
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "lz4",
                        "clevel": 5,
                        "shuffle": "bitshuffle",
                    },
                }
            ]
        elif compression == "zstd":
            chunk_codecs = [{"name": "zstd", "configuration": {"level": 3}}]
        elif compression == "none":
            chunk_codecs = []
        else:
            raise ValueError(f"Unknown compression: {compression}")

        codecs = chunk_codecs
        chunk_layout: dict[str, Any] = {"chunk": {"shape": list(chunks)}}
        if shards is not None:
            codecs = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": list(chunks),
                        "codecs": chunk_codecs,
                    },
                }
            ]
            chunk_layout = {"write_chunk": {"shape": list(shards)}}

        domain: dict[str, Any] = {"shape": list(shape)}
        if dimension_names:
            domain["labels"] = dimension_names

        try:
            dtype_str = dtype.name
        except AttributeError:
            dtype_str = str(dtype)

        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "schema": {
                "dtype": dtype_str,
                "domain": domain,
                "chunk_layout": chunk_layout,
                "codec": {"driver": "zarr3", "codecs": codecs},
            },
            "create": True,
            "delete_existing": overwrite,
        }
        return ts.open(spec).result()

    @staticmethod
    def _normalize_transform(
        values: Sequence[float] | None, ndim: int, fill: float
    ) -> list[float]:
        """Normalize transform vectors to match array dimensionality."""

        if values is None:
            return [fill] * ndim
        cast = [float(v) for v in values]
        if len(cast) == ndim:
            return cast
        if len(cast) == 3 and ndim >= 3:
            return [fill] * (ndim - 3) + cast
        return [fill] * ndim

    @staticmethod
    def _default_chunks(array: np.ndarray) -> list[int]:
        """Create sane default chunk sizes based on dimensionality."""

        if array.ndim == 2:
            return [int(array.shape[0]), int(array.shape[1])]
        if array.ndim == 3:
            return [1, int(array.shape[1]), int(array.shape[2])]
        if array.ndim == 4:
            return [1, 1, int(array.shape[2]), int(array.shape[3])]
        return list(array.shape)

    @staticmethod
    def _build_axes(v05: Any, ndim: int) -> list[Any]:
        """Build NGFF axes models for a given dimensionality."""

        axis_names = ["t", "c", "z", "y", "x"][-ndim:]
        axes: list[Any] = []
        for axis_name in axis_names:
            if axis_name in {"z", "y", "x"}:
                axes.append(v05.SpaceAxis(name=axis_name, unit="micrometer"))
            elif axis_name == "c":
                axes.append(v05.ChannelAxis(name="c"))
            else:
                axes.append(v05.TimeAxis(name="t", unit="second"))
        return axes

    @staticmethod
    def _entity_attributes_path(entity_root_path: Path | str) -> Path:
        """Path to per-entity metadata sidecar."""

        return Path(entity_root_path) / Path("attributes.json")

    @staticmethod
    def _image_store_path(image_path: Path | str) -> Path:
        """Normalize image path to the *.ome.zarr folder naming scheme."""

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
    def _write_extra_attributes(
        image_path: Path | str,
        extra_attributes: Mapping[str, Any],
        merge: bool = True,
    ) -> None:
        """Persist extra attributes directly into zarr.json."""

        if not extra_attributes:
            return

        image_root = qi2labDataStore._image_store_path(image_path)
        zarr_json_path = image_root / Path("zarr.json")
        data: dict[str, Any]
        if zarr_json_path.exists():
            data = qi2labDataStore._load_from_json(zarr_json_path)
        else:
            data = {}

        if merge:
            merged = {}
            current = data.get("extra_attributes")
            if isinstance(current, dict):
                merged.update(current)
            merged.update(dict(extra_attributes))
            data["extra_attributes"] = merged
        else:
            data["extra_attributes"] = dict(extra_attributes)

        qi2labDataStore._save_to_json(data, zarr_json_path)

    @staticmethod
    def _read_extra_attributes(image_path: Path | str) -> dict[str, Any]:
        """Load extra attributes from zarr.json."""

        image_root = qi2labDataStore._image_store_path(image_path)
        zarr_json_path = image_root / Path("zarr.json")
        data = qi2labDataStore._load_from_json(zarr_json_path)
        maybe_attrs = data.get("extra_attributes")
        if isinstance(maybe_attrs, dict):
            return maybe_attrs
        return {}

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        """Convert numpy/scalar containers to JSON-compatible values."""

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
    def _image_shape(image_path: Path | str) -> tuple[int, ...] | None:
        """Read image shape without loading all pixels."""

        path = qi2labDataStore._image_store_path(image_path)
        if not path.exists():
            return None

        open_group, _, _ = qi2labDataStore._import_yaozarrs()
        try:
            group = open_group(str(path))
            array_0 = group["0"]
            shape = getattr(array_0, "shape", None)
            if shape is None:
                shape = array_0.to_tensorstore().shape
            return tuple(int(dim) for dim in shape)
        except Exception:
            return None

    def _load_entity_attributes(
        self,
        entity_root_path: Path | str,
        image_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Load entity metadata from sidecar + image extra_attributes."""

        entity_root = Path(entity_root_path)
        merged = self._load_from_json(self._entity_attributes_path(entity_root))
        if not isinstance(merged, dict):
            merged = {}

        default_images = (
            "corrected_data",
            "registered_decon_data",
            f"registered_{self.feature_predictor_folder_name}_data",
            "opticalflow_xform_px",
        )
        candidate_names = image_names if image_names is not None else default_images
        for image_name in candidate_names:
            image_path = self._image_store_path(entity_root / Path(image_name))
            if not image_path.exists():
                continue
            extra_attrs = self._read_extra_attributes(image_path)
            if isinstance(extra_attrs, dict):
                merged.update(extra_attrs)

        return merged

    def _save_entity_attributes(
        self,
        entity_root_path: Path | str,
        updates: Mapping[str, Any],
        target_image_name: str | None = None,
        image_names: Sequence[str] | None = None,
    ) -> None:
        """Save metadata to image extra_attributes and entity sidecar."""

        if not updates:
            return

        entity_root = Path(entity_root_path)
        payload = {
            str(k): self._to_json_compatible(v) for k, v in dict(updates).items()
        }

        candidate_names: list[str] = []
        if target_image_name is not None:
            candidate_names.append(target_image_name)
        if image_names is not None:
            candidate_names.extend(
                [name for name in image_names if name not in candidate_names]
            )
        candidate_names.extend(
            [
                "corrected_data",
                "registered_decon_data",
                f"registered_{self.feature_predictor_folder_name}_data",
                "opticalflow_xform_px",
            ]
        )

        target_image_path: Path | None = None
        for image_name in candidate_names:
            image_path = self._image_store_path(entity_root / Path(image_name))
            if image_path.exists():
                target_image_path = image_path
                break

        if target_image_path is not None:
            self._write_extra_attributes(
                image_path=target_image_path, extra_attributes=payload, merge=True
            )

        sidecar_path = self._entity_attributes_path(entity_root)
        sidecar_attrs = self._load_from_json(sidecar_path)
        if not isinstance(sidecar_attrs, dict):
            sidecar_attrs = {}
        sidecar_attrs.update(payload)
        self._save_to_json(sidecar_attrs, sidecar_path)

    def _build_image_write_spec(
        self,
        dtype: str | None = None,
        stage_zyx_um: Sequence[float] | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build write spec with OME transforms and extra attributes."""

        spec = self._zarrv2_spec.copy()
        spec["metadata"] = dict(self._zarrv2_spec.get("metadata", {}))
        if dtype is not None:
            spec["metadata"]["dtype"] = dtype

        voxel_size = getattr(self, "_voxel_size_zyx_um", None)
        if voxel_size is not None:
            spec["ome_scale"] = [float(v) for v in np.asarray(voxel_size).tolist()]
        if stage_zyx_um is not None:
            spec["ome_translation"] = [float(v) for v in stage_zyx_um]
        if extra_attributes:
            spec["extra_attributes"] = {
                str(k): self._to_json_compatible(v)
                for k, v in dict(extra_attributes).items()
            }
        return spec

    def _resolve_original_tile_position_zyx_um(
        self,
        tile_id: str,
        round_id: str | None = None,
        bit_id: str | None = None,
    ) -> list[float] | None:
        """Resolve original tile stage position used for OME translation."""

        if round_id is not None:
            fiducial_entity = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            attrs = self._load_entity_attributes(fiducial_entity)
            stage = attrs.get("stage_zyx_um")
            if stage is not None:
                return [float(v) for v in stage]

        if bit_id is not None:
            round_linker = self.load_local_round_linker(tile=tile_id, bit=bit_id)
            if round_linker is not None and int(round_linker) > 0:
                linked_round_id = self._round_ids[int(round_linker) - 1]
                fiducial_entity = (
                    self._fiducial_root_path / Path(tile_id) / Path(linked_round_id)
                )
                attrs = self._load_entity_attributes(fiducial_entity)
                stage = attrs.get("stage_zyx_um")
                if stage is not None:
                    return [float(v) for v in stage]

        if getattr(self, "_round_ids", None):
            fiducial_entity = (
                self._fiducial_root_path / Path(tile_id) / Path(self._round_ids[0])
            )
            attrs = self._load_entity_attributes(fiducial_entity)
            stage = attrs.get("stage_zyx_um")
            if stage is not None:
                return [float(v) for v in stage]
        return None

    def _validate_core_image_shape(
        self,
        entity_root_path: Path | str,
        image_name: str,
        image: ArrayLike,
    ) -> None:
        """Enforce corrected/registered/feature-predictor image shape consistency."""

        entity_root = Path(entity_root_path)
        shape = tuple(int(v) for v in np.asarray(image).shape)
        required_names = {
            "corrected_data",
            "registered_decon_data",
            f"registered_{self.feature_predictor_folder_name}_data",
        }
        for candidate_name in required_names:
            if candidate_name == image_name:
                continue
            candidate_shape = self._image_shape(entity_root / Path(candidate_name))
            if candidate_shape is None:
                continue
            if tuple(candidate_shape) != shape:
                raise ValueError(
                    f"Image shape mismatch in {entity_root.name}: "
                    f"{image_name}={shape} but {candidate_name}={candidate_shape}. "
                    "corrected_data, registered_decon_data, and "
                    "registered_feature_predictor_data must match."
                )

    @staticmethod
    def _load_from_json(dictionary_path: Path | str) -> dict:
        """Load json as dictionary.

        Parameters
        ----------
        dictionary_path : Union[Path, str]
            Path to json file.

        Returns
        -------
        dictionary : dict
            Dictionary from json file.
        """

        try:
            with open(dictionary_path) as f:
                dictionary = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            dictionary = {}
        return dictionary

    @staticmethod
    def _save_to_json(dictionary: dict, dictionary_path: Path | str) -> None:
        """Save dictionary to json.

        Parameters
        ----------
        dictionary : dict
            The data to be saved.
        dictionary_path : Union[Path,str]
            The path to the JSON file where the data will be saved.
        """

        with open(dictionary_path, "w") as file:
            json.dump(dictionary, file, indent=4)

    @staticmethod
    def _load_from_microjson(dictionary_path: Path | str) -> dict:
        """Load cell outlines outlines microjson as dictionary.

        Parameters
        ----------
        dictionary_path : Union[Path, str]
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
    def _check_for_zarr_array(kvstore: Path | str, spec: dict) -> None:
        """Check if image exists and is readable via yaozarrs.

        Parameters
        ----------
        kvstore : Union[Path, str]
            Datastore location.
        spec : dict
            Zarr specification.
        """

        del spec
        open_group, _, _ = qi2labDataStore._import_yaozarrs()
        image_path = qi2labDataStore._extract_local_path_from_kvstore(kvstore)
        image_path = qi2labDataStore._image_store_path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        group = open_group(str(image_path))
        _ = group["0"]

    @staticmethod
    def _load_from_zarr_array(
        kvstore: dict, spec: dict, return_future: bool = True
    ) -> ArrayLike:
        """Read image data via yaozarrs.

        Defaults to returning future result.

        Parameters
        ----------
        kvstore : dict
            Tensorstore kvstore specification.
        spec : dict
            Tensorstore zarr specification.
        return_future : bool
            Return future (True) or immediately read (False).

        Returns
        -------
        array : ArrayLike
            Delayed (future) or immediate array.
        """

        del spec
        open_group, _, _ = qi2labDataStore._import_yaozarrs()
        image_path = qi2labDataStore._extract_local_path_from_kvstore(kvstore)
        image_path = qi2labDataStore._image_store_path(image_path)
        group = open_group(str(image_path))
        current_array = group["0"].to_tensorstore()

        read_future = current_array.read()
        return read_future if return_future else read_future.result()

    @staticmethod
    def _save_to_zarr_array(
        array: ArrayLike,
        kvstore: dict,
        spec: dict,
        return_future: bool | None = False,
    ) -> ArrayLike | None:
        """Save image data as OME-Zarr v0.5 using yaozarrs tensorstore writer.

        Defaults to returning future result.

        Parameters
        ----------
        array : ArrayLike
            Array to save.
        kvstore : dict
            Tensorstore kvstore specification.
        spec : dict
            Tensorstore zarr specification.
        return_future : Optional[bool]
            Return future (True) or immediately write (False).

        Returns
        -------
        write_future : Optional[ArrayLike]
            Delayed (future) if return_future is True.
        """

        open_group, v05, write_image = qi2labDataStore._import_yaozarrs()
        image_path = qi2labDataStore._extract_local_path_from_kvstore(kvstore)
        image_path = qi2labDataStore._image_store_path(image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        image_array = np.asarray(array)
        if image_array.dtype == np.float64:
            image_array = image_array.astype(np.float32)

        if image_array.ndim < 2 or image_array.ndim > 5:
            print(f"Unsupported array ndim for image write: {image_array.ndim}")
            return None

        metadata = spec.get("metadata", {}) if isinstance(spec, dict) else {}
        chunks = metadata.get("chunks")
        if chunks is None or len(chunks) != image_array.ndim:
            chunks = qi2labDataStore._default_chunks(image_array)
        compressor = (
            metadata.get("compressor", {}) if isinstance(metadata, dict) else {}
        )
        compression = "blosc-zstd"
        if isinstance(compressor, dict):
            cname = str(compressor.get("cname", "zstd")).lower()
            if cname == "lz4":
                compression = "blosc-lz4"
            elif cname == "zstd":
                compression = "blosc-zstd"

        scale = qi2labDataStore._normalize_transform(
            spec.get("ome_scale") if isinstance(spec, dict) else None,
            image_array.ndim,
            1.0,
        )
        translation = qi2labDataStore._normalize_transform(
            spec.get("ome_translation") if isinstance(spec, dict) else None,
            image_array.ndim,
            0.0,
        )
        extra_attributes = (
            spec.get("extra_attributes", {}) if isinstance(spec, dict) else {}
        )

        axes = qi2labDataStore._build_axes(v05, image_array.ndim)
        transforms = [
            v05.ScaleTransformation(scale=scale),
            v05.TranslationTransformation(translation=translation),
        ]
        datasets = [v05.Dataset(path="0", coordinateTransformations=transforms)]
        multiscales = [v05.Multiscale(axes=axes, datasets=datasets)]

        image_metadata = v05.Image(multiscales=multiscales)

        try:
            chunk_spec: tuple[int, ...] | str | None
            if chunks is None:
                chunk_spec = "auto"
            else:
                chunk_spec = tuple(int(c) for c in chunks)
            write_image(
                dest=str(image_path),
                image=image_metadata,
                datasets=image_array,
                extra_attributes=(dict(extra_attributes) if extra_attributes else None),
                writer=qi2labDataStore._create_array_tensorstore_qi2lab,
                overwrite=True,
                chunks=chunk_spec,
                compression=compression,
            )
            if extra_attributes:
                qi2labDataStore._write_extra_attributes(
                    image_path=image_path,
                    extra_attributes=extra_attributes,
                    merge=True,
                )
        except (OSError, TimeoutError, ValueError) as exc:
            print(exc)
            print("Error writing OME-Zarr array.")
            return None

        if return_future:
            group = open_group(str(image_path))
            current_array = group["0"].to_tensorstore()
            return current_array.read()
        return None

    @staticmethod
    def _load_from_parquet(parquet_path: Path | str) -> pd.DataFrame:
        """Load dataframe from parquet.

        Parameters
        ----------
        parquet_path : Union[Path, str]
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
        parquet_path : Union[Path, str]
            Path to parquet file.
        """

        df.to_parquet(parquet_path, engine="fastparquet", index=False)

    def _parse_datastore(self) -> None:
        """Parse datastore to discover available components."""

        # directory structure as defined by qi2lab spec
        self._datastore_state_json_path = self._datastore_path / Path(
            r"datastore_state.json"
        )
        # read in .json in root directory that indicates what steps have been run
        with open(self._datastore_state_json_path) as json_file:
            self._datastore_state = json.load(json_file)
        if float(self._datastore_state["Version"]) != 0.6:
            raise ValueError("Only datastore version 0.6 is supported by this build.")

        self.fiducial_folder_name = "fiducial"
        self.feature_predictor_folder_name = "feature_predictor"
        self._fiducial_root_path = self._datastore_path / Path(
            self.fiducial_folder_name
        )
        self._feature_predictor_localizations_root_path = self._datastore_path / Path(
            f"{self.feature_predictor_folder_name}_localizations"
        )
        self._calibrations_zarr_path = self._datastore_path / Path(r"calibrations")
        self._readouts_root_path = self._datastore_path / Path(r"readouts")
        self._decoded_root_path = self._datastore_path / Path(r"decoded")
        self._fused_root_path = self._datastore_path / Path(r"fused")
        self._segmentation_root_path = self._datastore_path / Path(r"segmentation")
        self._mtx_output_root_path = self._datastore_path / Path(r"mtx_output")

        # validate calibrations
        if self._datastore_state["Calibrations"]:
            if not (self._calibrations_zarr_path.exists()):
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
                elif key == "exp_order":
                    channel_list = [str(channel) for channel in attributes["channels_in_data"]]
                    self._experiment_order = pd.DataFrame(
                        attributes[key],
                        columns=channel_list,
                        dtype="int64",
                    )
                else:
                    setattr(self, "_" + key, attributes[key])

            self._tile_ids = [
                "tile" + str(tile_idx).zfill(4) for tile_idx in range(int(self._num_tiles))
            ]
            self._round_ids = [
                "round" + str(round_idx + 1).zfill(3)
                for round_idx in range(int(self._num_rounds))
            ]
            self._bit_ids = [
                "bit" + str(bit_idx + 1).zfill(3)
                for bit_idx in range(int(self._num_bits))
            ]

            psf_root_path = self._calibrations_zarr_path / Path("psf_data")
            try:
                if psf_root_path.exists():
                    psf_dirs = sorted(
                        [
                            entry
                            for entry in psf_root_path.iterdir()
                            if entry.is_dir()
                            and re.fullmatch(r"psf_\d{3}\.ome\.zarr", entry.name)
                        ],
                        key=lambda p: int(p.name[len("psf_") : len("psf_") + 3]),
                    )
                else:
                    psf_dirs = []

                if len(psf_dirs) > 0:
                    psf_list = []
                    for psf_dir in psf_dirs:
                        psf_array = self._load_from_zarr_array(
                            kvstore=self._get_kvstore_key(psf_dir),
                            spec=self._zarrv2_spec.copy(),
                            return_future=False,
                        )
                        psf_list.append(np.asarray(psf_array, dtype=np.float32))
                    self._psfs = psf_list
            except (OSError, ZarrError, ValueError, AttributeError):
                print("Calibration psfs missing.")

            # current_local_zarr_path = str(
            #     self._calibrations_zarr_path / Path("noise_map")
            # )

            # try:
            #     self._noise_map = (
            #         self._load_from_zarr_array(
            #             kvstore=self._get_kvstore_key(current_local_zarr_path),
            #             spec=self._zarrv2_spec,
            #         )
            #     ).result()
            # except Exception:
            #     print("Calibration noise map missing.")

        # validate fiducial and readout bits data
        if self._datastore_state["Corrected"]:
            if not (self._fiducial_root_path.exists()):
                raise FileNotFoundError("fiducial directory not initialized")
            else:
                fiducial_tile_ids = self._collect_strict_ids(
                    self._fiducial_root_path, prefix="tile", width=4
                )
                if len(fiducial_tile_ids) == 0:
                    raise FileNotFoundError("No tile folders found under fiducial/")
                current_tile_dir_path = self._fiducial_root_path / Path(
                    fiducial_tile_ids[0]
                )
                self._round_ids = self._collect_strict_ids(
                    current_tile_dir_path, prefix="round", width=3
                )
            if not (self._readouts_root_path.exists()):
                raise FileNotFoundError("Readout directory not initialized")
            else:
                readout_tile_ids = self._collect_strict_ids(
                    self._readouts_root_path, prefix="tile", width=4
                )
                if len(readout_tile_ids) == 0:
                    raise FileNotFoundError("No tile folders found under readouts/")
                current_tile_dir_path = self._readouts_root_path / Path(
                    readout_tile_ids[0]
                )
                self._bit_ids = self._collect_strict_ids(
                    current_tile_dir_path, prefix="bit", width=3
                )
            assert fiducial_tile_ids == readout_tile_ids, (
                "fiducial and readout tile ids do not match. Conversion error."
            )
            self._tile_ids = fiducial_tile_ids.copy()
            del fiducial_tile_ids, readout_tile_ids

            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
                attributes = self._load_entity_attributes(entity_root)

                keys_to_check = [
                    "stage_zyx_um",
                    "excitation_um",
                    "emission_um",
                    # "exposure_ms",
                    "psf_idx",
                ]

                for key in keys_to_check:
                    if key not in attributes.keys():
                        print(tile_id, round_id, key)
                        raise KeyError("Corrected fiducial attributes incomplete")
                if "bit_linker" not in attributes:
                    print(tile_id, round_id, "bit_linker")
                    raise KeyError("Corrected fiducial attributes incomplete")

                current_local_zarr_path = str(entity_root / Path("corrected_data"))

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
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

                current_local_zarr_path = str(entity_root / Path("corrected_data"))

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
                    print(tile_id, bit_id)
                    print("Corrected readout data missing.")

        # check and validate local registered data
        if self._datastore_state["LocalRegistered"]:
            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
                if round_id != self._round_ids[0]:
                    attributes = self._load_entity_attributes(entity_root)

                    keys_to_check = ["rigid_xform_xyz_px"]

                    for key in keys_to_check:
                        if key not in attributes.keys():
                            raise KeyError(
                                f"{round_id, tile_id} Rigid registration missing"
                            )

                    current_local_zarr_path = str(
                        entity_root / Path("opticalflow_xform_px")
                    )

                    try:
                        self._check_for_zarr_array(
                            self._get_kvstore_key(current_local_zarr_path),
                            self._zarrv2_spec.copy(),
                        )
                    except (OSError, ZarrError):
                        # print(tile_id, round_id)
                        # print("Optical flow registration data missing.")
                        pass

                current_local_zarr_path = str(
                    entity_root / Path("registered_decon_data")
                )
                if round_id == self._round_ids[0]:
                    try:
                        self._check_for_zarr_array(
                            self._get_kvstore_key(current_local_zarr_path),
                            self._zarrv2_spec.copy(),
                        )
                    except (OSError, ZarrError):
                        print(tile_id, round_id)
                        print("Registered fiducial data missing.")
                corrected_shape = self._image_shape(
                    entity_root / Path("corrected_data")
                )
                registered_shape = self._image_shape(
                    entity_root / Path("registered_decon_data")
                )
                if (
                    corrected_shape is not None
                    and registered_shape is not None
                    and corrected_shape != registered_shape
                ):
                    raise ValueError(
                        f"{tile_id} {round_id} corrected and registered shapes differ: "
                        f"{corrected_shape} != {registered_shape}"
                    )

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                entity_root = self._readouts_root_path / Path(tile_id) / Path(bit_id)
                current_local_zarr_path = str(
                    entity_root / Path("registered_decon_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
                    print(tile_id, bit_id)
                    print("Registered readout data missing.")

                current_local_zarr_path = str(
                    entity_root
                    / Path(f"registered_{self.feature_predictor_folder_name}_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
                    print(tile_id, bit_id)
                    print("Registered feature_predictor prediction missing.")
                corrected_shape = self._image_shape(
                    entity_root / Path("corrected_data")
                )
                registered_shape = self._image_shape(
                    entity_root / Path("registered_decon_data")
                )
                feature_shape = self._image_shape(
                    entity_root
                    / Path(f"registered_{self.feature_predictor_folder_name}_data")
                )
                shapes = [
                    shape
                    for shape in (corrected_shape, registered_shape, feature_shape)
                    if shape is not None
                ]
                if len(shapes) > 1 and any(shape != shapes[0] for shape in shapes[1:]):
                    raise ValueError(
                        f"{tile_id} {bit_id} corrected/registered/feature image shapes differ: "
                        f"{corrected_shape}, {registered_shape}, {feature_shape}"
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
        if self._datastore_state["GlobalRegistered"]:
            for tile_id in self._tile_ids:
                entity_root = (
                    self._fiducial_root_path / Path(tile_id) / Path(self._round_ids[0])
                )
                attributes = self._load_entity_attributes(entity_root)

                keys_to_check = ["affine_zyx_um", "origin_zyx_um", "spacing_zyx_um"]

                for key in keys_to_check:
                    if key not in attributes.keys():
                        raise KeyError("Global registration missing")

        # check and validate fused
        if self._datastore_state["Fused"]:
            fused_image_path = (
                self._fused_root_path
                / Path("fused.zarr")
                / Path(f"fused_{self.fiducial_folder_name}_iso_zyx")
            )
            attributes = self._read_extra_attributes(fused_image_path)

            keys_to_check = ["affine_zyx_um", "origin_zyx_um", "spacing_zyx_um"]

            for key in keys_to_check:
                if key not in attributes.keys():
                    raise KeyError("Fused image metadata missing")

            current_local_zarr_path = str(fused_image_path)

            try:
                self._check_for_zarr_array(
                    self._get_kvstore_key(current_local_zarr_path),
                    self._zarrv2_spec.copy(),
                )
            except (OSError, ZarrError):
                print("Fused data missing.")

        # check and validate cellpose segmentation
        if self._datastore_state["SegmentedCells"]:
            current_local_zarr_path = str(
                self._segmentation_root_path
                / Path("cellpose")
                / Path("cellpose.zarr")
                / Path(f"masks_{self.fiducial_folder_name}_iso_zyx")
            )

            try:
                self._check_for_zarr_array(
                    self._get_kvstore_key(current_local_zarr_path),
                    self._zarrv2_spec.copy(),
                )
            except (OSError, ZarrError):
                print("Cellpose data missing.")

            cell_outlines_path = (
                self._segmentation_root_path
                / Path("cellpose")
                / Path("imagej_rois")
                / Path("global_coords_rois.zip")
            )
            if not (cell_outlines_path.exists()):
                raise FileNotFoundError("Cellpose cell outlines missing.")

        # check and validate decoded spots
        if self._datastore_state["DecodedSpots"]:
            for tile_id in self._tile_ids:
                decoded_path = self._decoded_root_path / Path(
                    tile_id + "_decoded_features.parquet"
                )

                if not (decoded_path.exists()):
                    raise FileNotFoundError(tile_id + " decoded spots missing.")

        # check and validate filtered decoded spots
        if self._datastore_state["FilteredSpots"]:
            filtered_path = self._decoded_root_path / Path(
                "all_tiles_filtered_decoded_features.parquet"
            )

            if not (filtered_path.exists()):
                raise FileNotFoundError("filtered decoded spots missing.")

        if self._datastore_state["RefinedSpots"]:
            baysor_spots_path = (
                self._segmentation_root_path / Path("baysor") / Path("segmentation.csv")
            )

            if not (baysor_spots_path.exists()):
                raise FileNotFoundError("Baysor filtered decoded spots missing.")

        # check and validate mtx
        if self._datastore_state["mtxOutput"]:
            mtx_barcodes_path = self._mtx_output_root_path / Path("barcodes.tsv.gz")
            mtx_features_path = self._mtx_output_root_path / Path("features.tsv.gz")
            mtx_matrix_path = self._mtx_output_root_path / Path("matrix.tsv.gz")

            if (
                not (mtx_barcodes_path.exists())
                or not (mtx_features_path.exists())
                or not (mtx_matrix_path.exists())
            ):
                raise FileNotFoundError("mtx output missing.")

        try:
            self._baysor_path = Path(str(self._datastore_state["BaysorPath"]))
            self._baysor_options = Path(str(self._datastore_state["BaysorOptions"]))
            self._julia_threads = int(self._datastore_state["JuliaThreads"])
        except KeyError:
            self._baysor_path = r""
            self._baysor_options = r""
            self._julia_threads = 1

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
            data = getattr(self, "_codebook", None)

            if data is None:
                return None
            num_columns = len(data[0]) if data else 0
            columns = ["gene_id"] + [f"bit{i:02d}" for i in range(1, num_columns)]
            codebook_df = pd.DataFrame(data, columns=columns)

            gene_ids = codebook_df.iloc[:, 0].tolist()
            codebook_matrix = codebook_df.iloc[:, 1:].to_numpy().astype(int)
            del data, codebook_df
            return gene_ids, codebook_matrix
        except (KeyError, ValueError, TypeError):
            print("Error parsing codebook.")
            return None

    def initialize_tile(
        self,
        tile: int | str,
    ) -> None:
        """Initialize directory structure for a tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        """

        if getattr(self, "_experiment_order", None) is None:
            print("Assign experimental order before creating tiles.")
            return None

        if getattr(self, "_num_tiles", None) is None:
            print("Assign number of tiles before creating tiles.")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <" + str(self._num_tiles))
                return None
            else:
                tile_id = self._tile_ids[tile]
        elif isinstance(tile, str):
            if tile not in self._tile_ids:
                print("set valid tile id.")
                return None
            else:
                tile_id = tile
        else:
            print("'tile' must be integer index or string identifier")
            return None

        try:
            fiducial_tile_path = self._fiducial_root_path / Path(tile_id)
            fiducial_tile_path.mkdir()
            for round_idx, round_id in enumerate(self._round_ids):
                fiducial_round_path = fiducial_tile_path / Path(round_id)
                fiducial_round_path.mkdir()
                fiducial_round_attrs_path = self._entity_attributes_path(
                    fiducial_round_path
                )
                round_attrs = {
                    "bit_linker": self._experiment_order.to_numpy()[round_idx, 1:]
                    .astype(int)
                    .tolist(),
                }
                self._save_to_json(round_attrs, fiducial_round_attrs_path)
        except FileExistsError:
            print("Error creating fiducial tile. Does it exist already?")

        try:
            readout_tile_path = self._readouts_root_path / Path(tile_id)
            readout_tile_path.mkdir()
            for bit_idx, bit_id in enumerate(self._bit_ids):
                readout_bit_path = readout_tile_path / Path(bit_id)
                readout_bit_path.mkdir()
                readout_bit_attrs_path = self._entity_attributes_path(readout_bit_path)
                fiducial_channel = str(self._channels_in_data[0])
                readout_one_channel = str(self._channels_in_data[1])

                if len(self._channels_in_data) == 3:
                    readout_two_channel = str(self._channels_in_data[2])
                    condition_one = self._experiment_order[readout_one_channel] == (
                        bit_idx + 1
                    )
                    condition_two = self._experiment_order[readout_two_channel] == (
                        bit_idx + 1
                    )
                    combined_condition = condition_one | condition_two

                else:
                    combined_condition = self._experiment_order[
                        readout_one_channel
                    ] == (bit_idx + 1)
                matching_rows = self._experiment_order.loc[combined_condition]

                bit_attrs = {
                    "round_linker": int(matching_rows[fiducial_channel].values[0])
                }
                self._save_to_json(bit_attrs, readout_bit_attrs_path)
        except FileExistsError:
            print("Error creating readout tile. Does it exist already?")

    def load_local_bit_linker(
        self,
        tile: int | str,
        round: int | str,
    ) -> Sequence[int] | None:
        """Load readout bits linked to fidicual round for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.

        Returns
        -------
        bit_linker : Optional[Sequence[int]]
            Readout bits linked to fidicual round for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
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
            if round < 0:
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
        """Save readout bits linked to fidicual round for one tile.

        Parameters
        ----------
        bit_linker : Sequence[int]
            Readout bits linked to fidicual round for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
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
            if round < 0:
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
                entity_root_path=entity_root,
                updates={"bit_linker": values},
                target_image_name="corrected_data",
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
        """Load fidicual round linked to readout bit for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.

        Returns
        -------
        round_linker : Optional[Sequence[int]]
            Fidicual round linked to readout bit for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if bit < 0 or bit > len(self._bit_ids):
                print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
        """Save fidicual round linker attribute to readout bit for one tile.

        Parameters
        ----------
        round_linker : int
            Fidicual round linked to readout bit for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if bit < 0 or bit > len(self._bit_ids):
                print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={"round_linker": int(round_linker)},
                target_image_name="corrected_data",
            )
        except (TypeError, ValueError):
            print(tile_id, bit_id)
            print("Error writing round linker attribute.")
            return None

    def load_local_stage_position_zyx_um(
        self,
        tile: int | str,
        round: int | str,
    ) -> ArrayLike | None:
        """Load tile stage position for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.

        Returns
        -------
        stage_zyx_um : Optional[ArrayLike]
            Tile stage position for one tile.
        affine_zyx_um: Optional[ArrayLike]
            Affine transformation between stage and camera
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
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
            if round < 0:
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
            stage_zyx_um = attributes.get("stage_zyx_um")
            affine_zyx_px = attributes.get("affine_zyx_px")
            if stage_zyx_um is None or affine_zyx_px is None:
                print(tile_id, round_id)
                print("Stage position attribute not found.")
                return None
            return np.asarray(stage_zyx_um, dtype=np.float32), np.asarray(
                affine_zyx_px, dtype=np.float32
            )
        except (TypeError, ValueError):
            print(tile_id, round_id)
            print("Stage position attribute not found.")
            return None

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
            Tile stage position for one tile.
        affine_zyx_px: ArrayLike
            4x4 homogeneous affine matrix for stage transformation
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
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
            if round < 0:
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
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "stage_zyx_um": np.asarray(stage_zyx_um, dtype=np.float32).tolist(),
                    "affine_zyx_px": np.asarray(
                        affine_zyx_px, dtype=np.float32
                    ).tolist(),
                },
                target_image_name="corrected_data",
            )
        except (TypeError, ValueError):
            print(tile_id, round_id)
            print("Error writing stage position attribute.")
            return None

    def load_local_wavelengths_um(
        self,
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
    ) -> tuple[float, float] | None:
        """Load wavelengths for fidicual OR readout bit for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.

        Returns
        -------
        wavelengths_um : Optional[tuple[float, float]]
            Wavelengths for fidicual OR readout bit for one tile.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
                if round < 0:
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
        """Save wavelengths for fidicual OR readout bit for one tile.

        Parameters
        ----------
        wavelengths_um : tuple[float, float]
            Wavelengths for fidicual OR readout bit for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.

        Returns
        -------
        wavelengths_um : Optional[tuple[float, float]]
            Wavelengths for fidicual OR readout bit for one tile.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
                if round < 0:
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
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "excitation_um": float(wavelengths_um[0]),
                    "emission_um": float(wavelengths_um[1]),
                },
                target_image_name="corrected_data",
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
    ) -> ArrayLike | None:
        """Load gain and offset corrected image for fiducial OR readout bit for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        corrected_image : Optional[ArrayLike]
            Gain and offset corrected image for fiducial OR readout bit for one tile.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id)
                / Path("corrected_data")
            )
        else:
            if isinstance(round, int):
                if round < 0:
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
            current_local_zarr_path = str(
                self._fiducial_root_path
                / Path(tile_id)
                / Path(local_id)
                / Path("corrected_data")
            )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("Corrected image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<u2"
            corrected_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            return corrected_image
        except (OSError, ZarrError):
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
    ) -> None:
        """Save gain and offset corrected image.

        Parameters
        ----------
        image : ArrayLike
            Local corrected image.
        tile : Union[int, str]
            Tile index or tile id.
        gain_correction : bool
            Gain correction applied (True) or not (False).
        hotpixel_correction : bool
            Hotpixel correction applied (True) or not (False).
        shading_correction : bool
            Shading correction applied (True) or not (False).
        psf_idx : int
            PSF index.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.
        return_future : Optional[bool]
            Return future array.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
            current_local_zarr_path = entity_root / Path("corrected_data")
            stage_position = self._resolve_original_tile_position_zyx_um(
                tile_id=tile_id, bit_id=local_id
            )
        else:
            if isinstance(round, int):
                if round < 0:
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
            current_local_zarr_path = entity_root / Path("corrected_data")
            stage_position = self._resolve_original_tile_position_zyx_um(
                tile_id=tile_id, round_id=local_id
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
            spec = self._build_image_write_spec(
                dtype="<u2",
                stage_zyx_um=stage_position,
                extra_attributes=attributes,
            )
            self._save_to_zarr_array(
                image,
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates=attributes,
                target_image_name="corrected_data",
            )
        except (OSError, TimeoutError, ValueError) as e:
            print(e)
            print("Error saving corrected image.")
            return None

    def load_local_rigid_xform_xyz_px(
        self,
        tile: int | str,
        round: int | str,
    ) -> ArrayLike | None:
        """Load calculated rigid registration transform for one round and tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.

        Returns
        -------
        rigid_xform_xyz_px : Optional[ArrayLike]
            Local rigid registration transform for one round and tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        if isinstance(round, int):
            if round < 0:
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None
        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            attributes = self._load_entity_attributes(entity_root)
            rigid_xform_xyz_px = np.asarray(
                attributes["rigid_xform_xyz_px"], dtype=np.float32
            )
            return rigid_xform_xyz_px
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(tile_id, round_id)
            print("Rigid transform mapping back to first round not found.")
            return None

    def save_local_rigid_xform_xyz_px(
        self,
        rigid_xform_xyz_px: ArrayLike,
        tile: int | str,
        round: int | str,
    ) -> ArrayLike | None:
        """Save calculated rigid registration transform for one round and tile.

        Parameters
        ----------
        rigid_xform_xyz_px : ArrayLike
            Local rigid registration transform for one round and tile.
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.

        Returns
        -------
        rigid_xform_xyz_px : Optional[ArrayLike]
            Local rigid registration transform for one round and tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        if isinstance(round, int):
            if round < 0:
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None
        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "rigid_xform_xyz_px": np.asarray(
                        rigid_xform_xyz_px, dtype=np.float32
                    ).tolist()
                },
                target_image_name="registered_decon_data",
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            print("Error writing rigid transform attribute.")
            return None

    def load_coord_of_xform_px(
        self,
        tile: int | str | None,
        round: int | str | None,
        return_future: bool | None = True,
    ) -> tuple[ArrayLike, ArrayLike] | None:
        """Local fidicual optical flow matrix for one round and tile.

        Parameters
        ----------
        tile : Optional[Union[int, str]]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        of_xform_px : Optional[ArrayLike]
            Local fidicual optical flow matrix for one round and tile.
        downsampling : Optional[ArrayLike]
            Downsampling factor.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        if isinstance(round, int):
            if round < 0:
                print("Set round index >=0 and <" + str(self._num_rounds))
                return None
            else:
                round_id = self._round_ids[round]
        elif isinstance(round, str):
            if round not in self._round_ids:
                print("Set valid round id")
                return None
            else:
                round_id = round
        else:
            print("'round' must be integer index or string identifier")
            return None

        entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
        current_local_zarr_path = entity_root / Path("opticalflow_xform_px")

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("Optical flow transform mapping back to first round not found.")
            return None

        try:
            spec_of = self._build_image_write_spec(dtype="<f4")
            of_xform_px = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec_of,
                return_future,
            )
            attributes = self._load_entity_attributes(
                entity_root, image_names=("opticalflow_xform_px",)
            )
            block_size = np.asarray(attributes["block_size"], dtype=np.float32)
            block_stride = np.asarray(attributes["block_stride"], dtype=np.float32)

            return (of_xform_px, block_size, block_stride)
        except (OSError, ZarrError, KeyError) as e:
            print(e)
            print("Error loading optical flow transform.")
            return None

    def save_coord_of_xform_px(
        self,
        of_xform_px: ArrayLike,
        tile: int | str,
        block_size: Sequence[float],
        block_stride: Sequence[float],
        round: int | str,
        return_future: bool | None = False,
    ) -> None:
        """Save fidicual optical flow matrix for one round and tile.

        Parameters
        ----------
        of_xform_px : ArrayLike
            Local fidicual optical flow matrix for one round and tile.
        tile : Union[int, str]
            Tile index or tile id.
        block_size : Sequence[float]
            Block size for pixel warp
        block_stride: Sequence[float]
            Block stride for pixel warp
        round : Union[int, str]
            Round index or round id.
        return_future : Optional[bool]
            Return future array.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        if isinstance(round, int):
            if round < 0:
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
        current_local_zarr_path = entity_root / Path("opticalflow_xform_px")

        try:
            opticalflow_attrs = {
                "block_size": np.asarray(block_size, dtype=np.float32).tolist(),
                "block_stride": np.asarray(block_stride, dtype=np.float32).tolist(),
            }
            # Optical flow is a dense pixel-space field, so we do not encode
            # physical voxel scale or stage translation transforms here.
            spec_of = self._zarrv2_spec.copy()
            spec_of["metadata"] = dict(self._zarrv2_spec.get("metadata", {}))
            spec_of["metadata"]["dtype"] = "<f4"
            spec_of["extra_attributes"] = opticalflow_attrs
            self._save_to_zarr_array(
                of_xform_px,
                self._get_kvstore_key(current_local_zarr_path),
                spec_of,
                return_future,
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates=opticalflow_attrs,
                target_image_name="opticalflow_xform_px",
                image_names=("opticalflow_xform_px",),
            )
        except (OSError, TimeoutError):
            print("Error saving optical flow transform.")
            return None

    def load_local_registered_image(
        self,
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
        return_future: bool | None = True,
    ) -> ArrayLike | None:
        """Local registered, deconvolved image for fidiculial OR readout bit for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        registered_decon_image : Optional[ArrayLike]
            Registered, deconvolved image for fidiculial OR readout bit for one tile.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id)
                / Path("registered_decon_data")
            )
        else:
            if isinstance(round, int):
                if round < 0:
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
            current_local_zarr_path = str(
                self._fiducial_root_path
                / Path(tile_id)
                / Path(local_id)
                / Path("registered_decon_data")
            )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            # print("Registered deconvolved image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<u2"
            registered_decon_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            return registered_decon_image
        except (OSError, ZarrError) as e:
            print(e)
            print("Error loading registered deconvolved image.")
            return None

    def save_local_registered_image(
        self,
        registered_image: ArrayLike,
        tile: int | str,
        deconvolution: bool = True,
        round: int | str | None = None,
        bit: int | str | None = None,
        return_future: bool | None = False,
    ) -> None:
        """Save registered, deconvolved image.

        Parameters
        ----------
        registered_image : ArrayLike
            Registered, deconvolved image.
        tile : Union[int, str]
            Tile index or tile id.
        deconvolution : bool
            Deconvolution applied (True) or not (False).
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.
        return_future : Optional[bool]
            Return future array.
        """

        if (round is None and bit is None) or (round is not None and bit is not None):
            print("Provide either 'round' or 'bit', but not both")
            return None

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
            current_local_zarr_path = entity_root / Path("registered_decon_data")
            stage_position = self._resolve_original_tile_position_zyx_um(
                tile_id=tile_id, bit_id=local_id
            )
        else:
            if isinstance(round, int):
                if round < 0:
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
            current_local_zarr_path = entity_root / Path("registered_decon_data")
            stage_position = self._resolve_original_tile_position_zyx_um(
                tile_id=tile_id, round_id=local_id
            )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name="registered_decon_data",
                image=registered_image,
            )
            attributes = self._load_entity_attributes(entity_root)
            attributes["deconvolution"] = bool(deconvolution)
            spec = self._build_image_write_spec(
                dtype="<u2",
                stage_zyx_um=stage_position,
                extra_attributes=attributes,
            )
            self._save_to_zarr_array(
                registered_image,
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates=attributes,
                target_image_name="registered_decon_data",
            )
        except (OSError, TimeoutError, ValueError):
            print("Error saving corrected image.")
            return None

    def load_local_feature_predictor_image(
        self,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = True,
    ) -> ArrayLike | None:
        """Load readout bit feature_predictor prediction image for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.
        return_future : Optional[bool]
            return a future (true) or array (false)

        Returns
        -------
        registered_feature_predictor_image : Optional[ArrayLike]
            feature_predictor prediction image for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if bit < 0 or bit > len(self._bit_ids):
                print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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

        current_local_zarr_path = str(
            self._readouts_root_path
            / Path(tile_id)
            / Path(bit_id)
            / Path(f"registered_{self.feature_predictor_folder_name}_data")
        )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("feature_predictor prediction image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<f4"
            registered_feature_predictor_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            return registered_feature_predictor_image
        except (OSError, ZarrError) as e:
            print(e)
            print("Error loading feature_predictor image.")
            return None

    def save_local_feature_predictor_image(
        self,
        feature_predictor_image: ArrayLike,
        tile: int | str,
        bit: int | str,
        return_future: bool | None = False,
    ) -> None:
        """Save feature_predictor prediction image.

        Parameters
        ----------
        feature_predictor_image : ArrayLike
            feature_predictor prediction image.
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.
        return_future : Optional[bool]
            Return future array.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
                if bit < 0 or bit > len(self._bit_ids):
                    print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
                f"registered_{self.feature_predictor_folder_name}_data"
            )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name=f"registered_{self.feature_predictor_folder_name}_data",
                image=feature_predictor_image,
            )
            stage_position = self._resolve_original_tile_position_zyx_um(
                tile_id=tile_id, bit_id=local_id
            )
            attributes = self._load_entity_attributes(entity_root)
            spec = self._build_image_write_spec(
                dtype="<f4",
                stage_zyx_um=stage_position,
                extra_attributes=attributes,
            )
            self._save_to_zarr_array(
                feature_predictor_image,
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates=attributes,
                target_image_name=f"registered_{self.feature_predictor_folder_name}_data",
            )
        except (OSError, ZarrError, ValueError) as e:
            print(e)
            print("Error saving feature_predictor image.")
            return None

    def load_local_feature_predictor_spots(
        self,
        tile: int | str,
        bit: int | str,
    ) -> pd.DataFrame | None:
        """Load feature_predictor spot localizations and features for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.

        Returns
        -------
        feature_predictor_localizations : Optional[pd.DataFrame]
            feature_predictor localizations and features for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if bit < 0 or bit > len(self._bit_ids):
                print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if bit < 0 or bit > len(self._bit_ids):
                print("Set bit index >=0 and <=" + str(len(self._bit_ids)))
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

        if not (
            self._feature_predictor_localizations_root_path / Path(tile_id)
        ).exists():
            (self._feature_predictor_localizations_root_path / Path(tile_id)).mkdir()

        current_feature_predictor_localizations_path = (
            self._feature_predictor_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
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
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike] | None:
        """Load global registration transform for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.

        Returns
        -------
        affine_zyx_um : Optional[ArrayLike]
            Global affine registration transform for one tile.
        origin_zyx_um : Optional[ArrayLike]
            Global origin registration transform for one tile.
        spacing_zyx_um : Optional[ArrayLike]
            Global spacing registration transform for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        try:
            entity_root = (
                self._fiducial_root_path / Path(tile_id) / Path(self._round_ids[0])
            )
            attributes = self._load_entity_attributes(entity_root)
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)
            return (affine_zyx_um, origin_zyx_um, spacing_zyx_um)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(tile_id, self._round_ids[0])
            print("Global coordinate transforms not found")
            return None, None, None

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
            Global affine registration transform for one tile.
        origin_zyx_um : ArrayLike
            Global origin registration transform for one tile.
        spacing_zyx_um : ArrayLike
            Global spacing registration transform for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        """
        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        try:
            entity_root = (
                self._fiducial_root_path / Path(tile_id) / Path(self._round_ids[0])
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "affine_zyx_um": np.asarray(
                        affine_zyx_um, dtype=np.float32
                    ).tolist(),
                    "origin_zyx_um": np.asarray(
                        origin_zyx_um, dtype=np.float32
                    ).tolist(),
                    "spacing_zyx_um": np.asarray(
                        spacing_zyx_um, dtype=np.float32
                    ).tolist(),
                },
                target_image_name="registered_decon_data",
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(e)
            print("Could not save global coordinate transforms.")

    def load_global_fidicual_image(
        self,
        return_future: bool | None = True,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike] | None:
        """Load downsampled, fused fidicual image.

        Parameters
        ----------
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        fused_image : Optional[ArrayLike]
            Downsampled, fused fidicual image.
        affine_zyx_um : Optional[ArrayLike]
            Global affine registration transform for fused image.
        origin_zyx_um : Optional[ArrayLike]
            Global origin registration transform for fused image.
        spacing_zyx_um : Optional[ArrayLike]
            Global spacing registration transform for fused image.
        """

        current_local_zarr_path = (
            self._fused_root_path
            / Path("fused.zarr")
            / Path(f"fused_{self.fiducial_folder_name}_iso_zyx")
        )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("Globally registered, fused image not found.")
            return None

        try:
            fused_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            attributes = self._read_extra_attributes(image_path)
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)
            return fused_image, affine_zyx_um, origin_zyx_um, spacing_zyx_um
        except (OSError, ZarrError, KeyError):
            print("Error loading globally registered, fused image.")
            return None

    def save_global_fidicual_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        fusion_type: str = "fiducial",
        return_future: bool | None = False,
    ) -> None:
        """Save downsampled, fused fidicual image.

        Parameters
        ----------
        fused_image : ArrayLike
            Downsampled, fused fidicual image.
        affine_zyx_um : ArrayLike
            Global affine registration transform for fused image.
        origin_zyx_um : ArrayLike
            Global origin registration transform for fused image.
        spacing_zyx_um : ArrayLike
            Global spacing registration transform for fused image.
        fusion_type : str
            Type of fusion (fiducial or all_channels).
        return_future : Optional[bool]
            Return future array.
        """

        if fusion_type == "fiducial":
            filename = f"fused_{self.fiducial_folder_name}_iso_zyx"
        else:
            filename = "fused_all_channels_zyx"
        current_local_zarr_path = (
            self._fused_root_path / Path("fused.zarr") / Path(filename)
        )

        metadata_attrs = {
            "affine_zyx_um": np.asarray(affine_zyx_um, dtype=np.float32).tolist(),
            "origin_zyx_um": np.asarray(origin_zyx_um, dtype=np.float32).tolist(),
            "spacing_zyx_um": np.asarray(spacing_zyx_um, dtype=np.float32).tolist(),
        }
        try:
            spec = self._build_image_write_spec(
                dtype="<u2",
                extra_attributes=metadata_attrs,
            )
            self._save_to_zarr_array(
                fused_image.astype(np.uint16),
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
        except (OSError, TimeoutError):
            print("Error saving fused image.")
            return None

    def load_local_decoded_spots(
        self,
        tile: int | str,
    ) -> pd.DataFrame | None:
        """Load decoded spots and features for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.

        Returns
        -------
        tile_features : Optional[pd.DataFrame]
            Decoded spots and features for one tile.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        current_tile_features_path = self._decoded_root_path / Path(
            tile_id + "_decoded_features.parquet"
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
    ) -> None:
        """Save decoded spots and features for one tile.

        Parameters
        ----------
        features_df : pd.DataFrame
            Decoded spots and features for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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

        current_tile_features_path = self._decoded_root_path / Path(
            tile_id + "_decoded_features.parquet"
        )

        self._save_to_parquet(features_df, current_tile_features_path)

    def load_global_filtered_decoded_spots(
        self,
    ) -> pd.DataFrame | None:
        """Load all decoded and filtered spots.

        Returns
        -------
        all_tiles_filtered : Optional[pd.DataFrame]
            All decoded and filtered spots.
        """

        current_global_filtered_decoded_dir_path = self._datastore_path / Path(
            "all_tiles_filtered_decoded_features"
        )
        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("decoded_features.parquet")
        )

        if not current_global_filtered_decoded_path.exists():
            print("Global, filtered, decoded spots not found.")
            return None
        else:
            all_tiles_filtered = self._load_from_parquet(
                current_global_filtered_decoded_path
            )
            return all_tiles_filtered

    def save_global_filtered_decoded_spots(
        self,
        filtered_decoded_df: pd.DataFrame,
    ) -> None:
        """Save all decoded and filtered spots.

        Parameters
        ----------
        filtered_decoded_df : pd.DataFrame
            All decoded and filtered spots.
        """

        current_global_filtered_decoded_dir_path = self._datastore_path / Path(
            "all_tiles_filtered_decoded_features"
        )

        if not current_global_filtered_decoded_dir_path.exists():
            current_global_filtered_decoded_dir_path.mkdir()

        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("decoded_features.parquet")
        )

        self._save_to_parquet(filtered_decoded_df, current_global_filtered_decoded_path)

    def load_global_cellpose_outlines(
        self,
    ) -> dict | None:
        """Load Cellpose max projection cell outlines.

        Returns
        -------
        cellpose_outlines : Optional[dict]
            Cellpose cell mask outlines.
        """

        current_cellpose_outlines_path = (
            self._segmentation_root_path / Path("cellpose") / Path("cell_outlines.json")
        )

        if not current_cellpose_outlines_path.exists():
            print("Cellpose cell mask outlines not found.")
            return None
        else:
            cellpose_outlines = self._load_from_microjson(
                current_cellpose_outlines_path
            )
            return cellpose_outlines

    def load_global_cellpose_segmentation_image(
        self,
        return_future: bool | None = True,
    ) -> ArrayLike | None:
        """Load Cellpose max projection, downsampled segmentation image.

        Parameters
        ----------
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        fused_image : Optional[ArrayLike]
            Cellpose max projection, downsampled segmentation image.
        """

        current_local_zarr_path = (
            self._segmentation_root_path
            / Path("cellpose")
            / Path("cellpose.zarr")
            / Path(f"masks_{self.fiducial_folder_name}_iso_zyx")
        )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("Cellpose prediction on global fused image not found.")
            return None

        try:
            fused_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            return fused_image
        except (OSError, ZarrError):
            print("Error loading Cellpose image.")
            return None

    def save_global_cellpose_segmentation_image(
        self,
        cellpose_image: ArrayLike,
        downsampling: Sequence[float],
        return_future: bool | None = False,
    ) -> None:
        """Save Cellpose max projection, downsampled segmentation image.

        Parameters
        ----------
        cellpose_image : ArrayLike
            Cellpose max projection, downsampled segmentation image.
        downsampling : Sequence[float]
            Downsample factors.
        return_future : Optional[bool]
            Return future array.
        """

        current_local_zarr_path = (
            self._segmentation_root_path
            / Path("cellpose")
            / Path("cellpose.zarr")
            / Path(f"masks_{self.fiducial_folder_name}_iso_zyx")
        )

        attributes = {
            "downsampling": np.asarray(downsampling, dtype=np.float32).tolist()
        }

        try:
            spec = self._build_image_write_spec(extra_attributes=attributes)
            self._save_to_zarr_array(
                cellpose_image,
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
        except (OSError, TimeoutError):
            print("Error saving Cellpose image.")
            return None

    def save_spots_prepped_for_baysor(
        self, prepped_for_baysor_df: pd.DataFrame
    ) -> None:
        """Save spots prepped for Baysor.

        Parameters
        ----------
        prepped_for_baysor_df : pd.DataFrame
            Spots prepped for Baysor.
        """

        current_global_filtered_decoded_dir_path = self._datastore_path / Path(
            "all_tiles_filtered_decoded_features"
        )

        if not current_global_filtered_decoded_dir_path.exists():
            current_global_filtered_decoded_dir_path.mkdir()

        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("transcripts.parquet")
        )

        self._save_to_parquet(
            prepped_for_baysor_df, current_global_filtered_decoded_path
        )

    def run_baysor(self) -> None:
        """Run Baysor"

        Assumes that spots are prepped for Baysor and the Baysor path and options are set.
        Reformats ROIs into ImageJ style ROIs for later use.
        """

        import subprocess

        baysor_input_path = (
            self._datastore_path
            / Path("all_tiles_filtered_decoded_features")
            / Path("transcripts.parquet")
        )
        baysor_output_path = self._segmentation_root_path / Path("baysor")
        baysor_output_path.mkdir(exist_ok=True)

        julia_threading = r"JULIA_NUM_THREADS=" + str(self._julia_threads) + " "
        preview_baysor_options = r"preview -c " + str(self._baysor_options)
        command = (
            julia_threading
            + str(self._baysor_path)
            + " "
            + preview_baysor_options
            + " "
            + str(baysor_input_path)
            + " -o "
            + str(baysor_output_path)
        )

        try:
            result = subprocess.run(command, shell=True, check=True)
            print("Baysor finished with return code:", result.returncode)
        except subprocess.CalledProcessError as e:
            print("Baysor failed with:", e)

        # first try to run Baysor assuming that prior segmentations are present
        try:
            run_baysor_options = r"run -p -c " + str(self._baysor_options)
            command = (
                julia_threading
                + str(self._baysor_path)
                + " "
                + run_baysor_options
                + " "
                + str(baysor_input_path)
                + " -o "
                + str(baysor_output_path)
                + " --polygon-format GeometryCollectionLegacy --count-matrix-format tsv :cell_id"
            )
            result = subprocess.run(command, shell=True, check=True)
            print("Baysor finished with return code:", result.returncode)
        except subprocess.CalledProcessError:
            # then fall back and run without prior segmentations.
            # IMPORTANT: the .toml file has to be defined correctly for this to work!
            try:
                run_baysor_options = r"run -p -c " + str(self._baysor_options)
                command = (
                    julia_threading
                    + str(self._baysor_path)
                    + " "
                    + run_baysor_options
                    + " "
                    + str(baysor_input_path)
                    + " -o "
                    + str(baysor_output_path)
                    + " --count-matrix-format tsv"
                )
                result = subprocess.run(command, shell=True, check=True)
                print("Baysor finished with return code:", result.returncode)
            except subprocess.CalledProcessError as e:
                print("Baysor failed with:", e)

    def reformat_baysor_3D_oultines(self) -> None:
        """Reformat baysor 3D json file into ImageJ ROIs."""
        import re

        # Load the JSON file
        baysor_output_path = self._segmentation_root_path / Path("baysor")
        baysor_segmentation = baysor_output_path / Path(
            r"segmentation_polygons_3d.json"
        )
        with open(baysor_segmentation) as file:
            data = json.load(file)

        # Dictionary to group polygons by cell ID
        cell_polygons = defaultdict(list)

        def parse_z_range(z_range):  # noqa
            cleaned_range = re.sub(
                r"[^\d.,-]", "", z_range
            )  # Remove non-numeric, non-period, non-comma, non-dash characters
            return map(float, cleaned_range.split(","))

        # Iterate through each z-plane and corresponding polygons
        for z_range, details in data.items():
            z_start, z_end = parse_z_range(z_range)

            for geometry in details["geometries"]:
                coordinates = geometry["coordinates"][
                    0
                ]  # Assuming the outer ring of the polygon
                cell_id = geometry["cell"]  # Get the cell ID

                # Store the polygon with its z-range
                cell_polygons[cell_id].append(
                    {"z_start": z_start, "z_end": z_end, "coordinates": coordinates}
                )

        rois = []

        # Process each cell ID to create 3D ROIs
        for cell_id, polygons in cell_polygons.items():
            for _idx, polygon in enumerate(polygons):
                x_coords = [point[0] for point in polygon["coordinates"]]
                y_coords = [point[1] for point in polygon["coordinates"]]

                z_start = polygon["z_start"]
                z_end = polygon["z_end"]

                try:
                    # Create an ImageJRoi object for the polygon using frompoints
                    coords = list(
                        zip(x_coords, y_coords, strict=False)
                    )  # List of (x, y) tuples
                    roi = ImagejRoi.frompoints(coords)
                    roi.roitype = ROI_TYPE.POLYGON  # Set the ROI type to Polygon
                    roi.coordinates = coords  # Explicitly assign coordinates to the ROI
                    roi.name = f"cell_{cell_id!s}_zstart_{z_start!s}_zend_{z_end!s}"  # Ensure unique name
                    rois.append(roi)
                except Exception as e:
                    print(f"Error while creating ROI for cell ID {cell_id}: {e}")

        # Write all ROIs to a ZIP file
        output_file = baysor_output_path / Path(r"3d_cell_rois.zip")
        roiwrite(output_file, rois, mode="w")

    def load_global_baysor_filtered_spots(
        self,
    ) -> pd.DataFrame | None:
        """Load Baysor re-assigned decoded RNA.

        Assumes Baysor has been run.

        Returns
        -------
        baysor_filtered_genes : Optional[pd.DataFrame]
            Baysor re-assigned decoded RNA.
        """

        current_baysor_spots_path = (
            self._segmentation_root_path / Path("baysor") / Path("segmentation.csv")
        )

        if not current_baysor_spots_path.exists():
            print("Baysor filtered genes not found.")
            return None
        else:
            baysor_filtered_genes = self._load_from_csv(current_baysor_spots_path)
            return baysor_filtered_genes

    def load_global_baysor_outlines(
        self,
    ) -> dict | None:
        """Load Baysor cell outlines.

        Assumes Baysor has been run.

        Returns
        -------
        baysor_outlines : Optional[dict]
            Baysor cell outlines.
        """

        current_baysor_outlines_path = (
            self._segmentation_root_path / Path("baysor") / Path(r"3d_cell_rois.zip")
        )

        if not current_baysor_outlines_path.exists():
            print("Baysor outlines not found.")
            return None
        else:
            baysor_rois = roiread(current_baysor_outlines_path)
            return baysor_rois

    @staticmethod
    def _roi_to_shapely(roi):  # noqa
        return Polygon(roi.subpixel_coordinates[:, ::-1])

    def reprocess_and_save_filtered_spots_with_baysor_outlines(self) -> None:
        """Reprocess filtered spots using baysor cell outlines, then save.

        Loads the 3D cell outlines from Baysor, checks all points to see what
        (if any) cell outline that the spot falls within, and then saves the
        data back to the datastore.
        """
        import re

        from rtree import index

        rois = self.load_global_baysor_outlines()
        filtered_spots_df = self.load_global_filtered_decoded_spots()

        parsed_spots_df = filtered_spots_df[
            [
                "gene_id",
                "global_z",
                "global_y",
                "global_x",
                "cell_id",
                "tile_idx",
            ]
        ].copy()
        parsed_spots_df.rename(
            columns={
                "global_x": "x",
                "global_y": "y",
                "global_z": "z",
                "gene_id": "gene",
                "cell_id": "cell",
            },
            inplace=True,
        )
        parsed_spots_df["transcript_id"] = pd.util.hash_pandas_object(
            parsed_spots_df, index=False
        )

        parsed_spots_df["assignment_confidence"] = 1.0

        # Create spatial index for ROIs
        roi_index = index.Index()
        roi_map = {}  # Map index IDs to ROIs

        for idx, roi in enumerate(rois):
            # Ensure roi.coordinates contains the polygon points
            coords = roi.coordinates()

            # Insert the polygon bounds into the spatial index
            polygon = Polygon(coords)
            roi_index.insert(idx, polygon.bounds)  # Use polygon bounds for indexing
            roi_map[idx] = roi

        # Function to check a single point
        def point_in_roi(row: pd.Series) -> str | float:
            """Determine if point is within ROI.

            Parameters
            ----------
            row: pd.Series
                current point to check

            Returns
            -------
            name: str | float
                cell_id or -1
            """
            point = Point(row["x"], row["y"])
            candidate_indices = list(
                roi_index.intersection(point.bounds)
            )  # Search spatial index
            for idx in candidate_indices:
                roi = roi_map[idx]
                match = re.search(r"zstart_([-\d.]+)_zend_([-\d.]+)", roi.name)
                if match:
                    z_start = float(match.group(1))
                    z_end = float(match.group(2))
                    if z_start <= row["z"] <= z_end:
                        polygon = Polygon(roi.coordinates())
                        if polygon.contains(point):
                            return str(roi.name.split("_")[1])
            return -1

        # Apply optimized spatial lookup
        parsed_spots_df["cell"] = parsed_spots_df.apply(point_in_roi, axis=1)
        parsed_spots_df = parsed_spots_df.loc[parsed_spots_df["cell"] != -1]

        current_global_filtered_decoded_path = (
            self._datastore_path
            / Path("all_tiles_filtered_decoded_features")
            / Path("refined_transcripts.parquet")
        )

        self._save_to_parquet(parsed_spots_df, current_global_filtered_decoded_path)

    def save_mtx(self, spots_source: str = "baysor") -> None:
        """Save mtx file for downstream analysis. Assumes Baysor has been run.

        Parameters
        ----------
        spots_source: str, default "baysor"
            source of spots. "baysor" or "resegmented".
        """

        from merfish3danalysis.utils.dataio import create_mtx

        if spots_source == "baysor":
            spots_path = (
                self._datastore_path
                / Path("segmentation")
                / Path("baysor")
                / Path("segmentation.csv")
            )
        elif spots_source == "resegmented":
            spots_path = (
                self._datastore_path
                / Path("all_tiles_filtered_decoded_features")
                / Path("refined_transcripts.parquet")
            )

        mtx_output_path = self._datastore_path / Path("mtx_output")

        create_mtx(
            spots_path=spots_path,
            output_dir_path=mtx_output_path,
        )
