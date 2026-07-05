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
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import TimeoutError
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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
    validate : bool, default True
        Validate datastore contents on open. Set False to skip expensive
        zarr readability checks and load only metadata/path structure.

    """

    def __init__(self, datastore_path: str | Path, validate: bool = True) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        datastore_path : str | Path
            Function argument.
        validate : bool
            Function argument.
        """
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
            self._parse_datastore(validate=validate)
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
        """
        Path to calibrations metadata sidecar.

        Returns
        -------
        Path
            Function result.
        """

        return self._calibrations_zarr_path / Path("attributes.json")

    def _load_calibrations_attributes(self) -> dict[str, Any]:
        """
        Load calibrations metadata sidecar.

        Returns
        -------
        dict[str, Any]
            Function result.
        """

        attributes = self._load_from_json(self._calibrations_attributes_path())
        if not isinstance(attributes, dict):
            raise ValueError("calibrations/attributes.json is invalid.")
        return attributes

    def _save_calibrations_attributes(self, attributes: Mapping[str, Any]) -> None:
        """
        Persist calibrations metadata sidecar.

        Parameters
        ----------
        attributes : Mapping[str, Any]
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        self._save_to_json(
            {str(k): self._to_json_compatible(v) for k, v in dict(attributes).items()},
            self._calibrations_attributes_path(),
        )

    def _set_calibration_attribute(self, key: str, value: Any) -> None:
        """
        Update one calibration metadata field.

        Parameters
        ----------
        key : str
            Function argument.
        value : Any
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        attributes = self._load_calibrations_attributes()
        attributes[str(key)] = self._to_json_compatible(value)
        self._save_calibrations_attributes(attributes)

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
            Metadata are written to ``calibrations/attributes.json``.
        """

        self._set_calibration_attribute(
            "chromatic_affine_transforms_zyx_um",
            calibration,
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
            attributes = self._load_calibrations_attributes()
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {}
        calibration = attributes.get("chromatic_affine_transforms_zyx_um", {})
        if isinstance(calibration, dict):
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

        calibration = self.load_chromatic_affine_transforms_zyx_um()
        channels = calibration.get("channels", {})
        if not isinstance(channels, Mapping):
            return np.eye(4, dtype=np.float32)

        candidates = []
        if channel_name is not None:
            channel = channels.get(str(channel_name))
            if isinstance(channel, Mapping):
                candidates.append(channel)
        if channel_index is not None:
            for channel in channels.values():
                if isinstance(channel, Mapping) and int(
                    channel.get("channel_index", -1)
                ) == int(channel_index):
                    candidates.append(channel)
        if wavelength_um is not None:
            wavelength = float(wavelength_um)
            for channel in channels.values():
                if not isinstance(channel, Mapping):
                    continue
                stored = channel.get("wavelength_um")
                if stored is not None and np.isclose(float(stored), wavelength):
                    candidates.append(channel)

        for channel in candidates:
            affine = channel.get("affine_zyx_um")
            if affine is not None:
                return np.asarray(affine, dtype=np.float32)
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def _strict_id_sort_key(name: str, prefix: str, width: int) -> int:
        """
        Validate and parse strict zero-padded identifiers.

        Parameters
        ----------
        name : str
            Function argument.
        prefix : str
            Function argument.
        width : int
            Function argument.

        Returns
        -------
        int
            Function result.
        """

        match = re.fullmatch(rf"{re.escape(prefix)}(\d{{{width}}})", name)
        if match is None:
            raise ValueError(
                f"Invalid identifier '{name}'. Expected '{prefix}' followed by {width} digits."
            )
        return int(match.group(1))

    @classmethod
    def _collect_strict_ids(cls, parent: Path, prefix: str, width: int) -> list[str]:
        """
        Collect and sort strict identifiers under a folder.

        Parameters
        ----------
        parent : Path
            Function argument.
        prefix : str
            Function argument.
        width : int
            Function argument.

        Returns
        -------
        list[str]
            Function result.
        """

        def sort_id(value: str) -> tuple[int, int, str]:
            return cls._strict_id_sort_key(value, prefix, width)

        ids = [entry.name for entry in parent.iterdir() if entry.is_dir()]
        ids.sort(key=sort_id)
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
            Function argument.

        Returns
        -------
        pd.DataFrame
            Function result.
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
        value : Union[ArrayLike, pd.DataFrame]
            New round and bit order.
        """

        self._experiment_order = self._coerce_experiment_order_dataframe(value)

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
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", decode_run_key):
            raise ValueError(
                "decode_run_key may only contain letters, numbers, '.', '_', and '-'."
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
        return self._decoded_root_path / Path(decode_run_key)

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

        root = self._decoded_run_root(decode_run_key) / Path("temporary")
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

        root = self._datastore_path / Path("all_tiles_filtered_decoded_features")
        decode_run_key = self._validate_decode_run_key(decode_run_key)
        if decode_run_key is None:
            return root
        return root / Path(decode_run_key)

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
        calib_attrs = self._load_calibrations_attributes()
        run_attrs = calib_attrs.get("decode_normalization_runs", {}).get(
            decode_run_key, {}
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
        """

        if kind == "global":
            if decode_run_key is None:
                self.global_normalization_vector = normalization_vector
                self.global_background_vector = background_vector
                return
            norm_key = "global_normalization_vector"
            background_key = "global_background_vector"
        elif kind == "iterative":
            if decode_run_key is None:
                self.iterative_normalization_vector = normalization_vector
                self.iterative_background_vector = background_vector
                return
            norm_key = "iterative_normalization_vector"
            background_key = "iterative_background_vector"
        else:
            raise ValueError("kind must be one of 'global' or 'iterative'.")

        decode_run_key = self._validate_decode_run_key(decode_run_key)
        calib_attrs = self._load_calibrations_attributes()
        runs = dict(calib_attrs.get("decode_normalization_runs", {}))
        run_attrs = dict(runs.get(decode_run_key, {}))
        if decode_mode is not None:
            run_attrs["decode_mode"] = str(decode_mode)
        run_attrs[norm_key] = np.asarray(normalization_vector, dtype=np.float32)
        run_attrs[background_key] = np.asarray(background_vector, dtype=np.float32)
        runs[decode_run_key] = run_attrs
        calib_attrs["decode_normalization_runs"] = runs
        self._save_calibrations_attributes(calib_attrs)

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
        """
        Import yaozarrs lazily so module import remains lightweight.

        Returns
        -------
        tuple[Any, Any, Any]
            Function result.
        """

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
        """
        Extract a local filesystem path from a kvstore-like input.

        Parameters
        ----------
        kvstore : dict | Path | str
            Function argument.

        Returns
        -------
        Path
            Function result.
        """

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
        """
        Create zarr3 arrays with qi2lab compression defaults via tensorstore.

        Parameters
        ----------
        path : Path
            Function argument.
        shape : tuple[int, ...]
            Function argument.
        dtype : Any
            Function argument.
        chunks : tuple[int, ...]
            Function argument.
        shards : tuple[int, ...] | None
            Function argument.
        dimension_names : list[str] | None
            Function argument.
        overwrite : bool
            Function argument.
        compression : str
            Function argument.

        Returns
        -------
        Any
            Function result.
        """

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
        """
        Normalize transform vectors to match array dimensionality.

        Parameters
        ----------
        values : Sequence[float] | None
            Function argument.
        ndim : int
            Function argument.
        fill : float
            Function argument.

        Returns
        -------
        list[float]
            Function result.
        """

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
        """
        Create sane default chunk sizes based on dimensionality.

        Parameters
        ----------
        array : np.ndarray
            Function argument.

        Returns
        -------
        list[int]
            Function result.
        """

        if array.ndim == 2:
            return [int(array.shape[0]), int(array.shape[1])]
        if array.ndim == 3:
            return [1, int(array.shape[1]), int(array.shape[2])]
        if array.ndim == 4:
            return [1, 1, int(array.shape[2]), int(array.shape[3])]
        return list(array.shape)

    @staticmethod
    def _fused_image_chunks(array: np.ndarray) -> list[int]:
        """
        Create chunk sizes tailored for large fused images only.

        Parameters
        ----------
        array : np.ndarray
            Function argument.

        Returns
        -------
        list[int]
            Function result.
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
        return qi2labDataStore._default_chunks(array)

    @staticmethod
    def _build_axes(v05: Any, ndim: int) -> list[Any]:
        """
        Build NGFF axes models for a given dimensionality.

        Parameters
        ----------
        v05 : Any
            Function argument.
        ndim : int
            Function argument.

        Returns
        -------
        list[Any]
            Function result.
        """

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
        """
        Path to per-entity metadata sidecar.

        Parameters
        ----------
        entity_root_path : Path | str
            Function argument.

        Returns
        -------
        Path
            Function result.
        """

        return Path(entity_root_path) / Path("attributes.json")

    @staticmethod
    def _image_store_path(image_path: Path | str) -> Path:
        """
        Normalize image path to the *.ome.zarr folder naming scheme.

        Parameters
        ----------
        image_path : Path | str
            Function argument.

        Returns
        -------
        Path
            Function result.
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
    def _read_extra_attributes(image_path: Path | str) -> dict[str, Any]:
        """
        Load extra attributes through yaozarrs.

        Parameters
        ----------
        image_path : Path | str
            Function argument.

        Returns
        -------
        dict[str, Any]
            Function result.
        """

        image_root = qi2labDataStore._image_store_path(image_path)
        open_group, _, _ = qi2labDataStore._import_yaozarrs()
        attrs = dict(open_group(str(image_root)).attrs)
        attrs.pop("ome", None)
        return attrs

    @staticmethod
    def _write_extra_attributes(
        image_path: Path | str,
        extra_attributes: Mapping[str, Any],
        merge: bool = True,
    ) -> None:
        """
        Write image-level extra attributes for externally-created OME-Zarr stores.

        Parameters
        ----------
        image_path : Path | str
            Image group path. Both the directory path and the corresponding
            OME-Zarr store path are accepted.
        extra_attributes : Mapping[str, Any]
            Attribute updates to write at the image root.
        merge : bool, default=True
            If True, merge updates into existing attributes. If False, replace
            the existing image-level attributes with ``extra_attributes``.

        Returns
        -------
        None
            Attributes are written to ``zarr.json`` for Zarr v3 stores or to
            ``.zattrs`` for Zarr v2 stores.
        """

        image_root = qi2labDataStore._image_store_path(image_path)
        payload = {
            str(k): qi2labDataStore._to_json_compatible(v)
            for k, v in dict(extra_attributes).items()
        }

        zarr_json_path = image_root / Path("zarr.json")
        if zarr_json_path.exists():
            with zarr_json_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if merge:
                attributes = metadata.get("attributes", {})
                if not isinstance(attributes, dict):
                    attributes = {}
                attributes.update(payload)
                metadata["attributes"] = attributes
            else:
                metadata["attributes"] = payload
            with zarr_json_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            return

        zattrs_path = image_root / Path(".zattrs")
        if zattrs_path.exists() and merge:
            with zattrs_path.open("r", encoding="utf-8") as handle:
                attributes = json.load(handle)
            if not isinstance(attributes, dict):
                attributes = {}
        else:
            attributes = {}
        attributes.update(payload)
        with zattrs_path.open("w", encoding="utf-8") as handle:
            json.dump(attributes, handle, indent=2)

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        """
        Convert numpy/scalar containers to JSON-compatible values.

        Parameters
        ----------
        value : Any
            Function argument.

        Returns
        -------
        Any
            Function result.
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
    def _image_shape(image_path: Path | str) -> tuple[int, ...] | None:
        """
        Read image shape without loading all pixels.

        Parameters
        ----------
        image_path : Path | str
            Function argument.

        Returns
        -------
        tuple[int, ...] | None
            Function result.
        """

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
        """
        Load entity metadata from sidecar + image extra_attributes.

        Parameters
        ----------
        entity_root_path : Path | str
            Function argument.
        image_names : Sequence[str] | None
            Function argument.

        Returns
        -------
        dict[str, Any]
            Function result.
        """

        entity_root = Path(entity_root_path)
        merged: dict[str, Any] = {}

        default_images = (
            "corrected_data",
            "registered_decon_data",
            "decon_data",
            f"{self.feature_predictor_folder_name}_data",
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

        sidecar_attrs = self._load_from_json(self._entity_attributes_path(entity_root))
        if isinstance(sidecar_attrs, dict):
            merged.update(sidecar_attrs)

        return merged

    def _save_entity_attributes(
        self,
        entity_root_path: Path | str,
        updates: Mapping[str, Any],
        target_image_name: str | None = None,
        image_names: Sequence[str] | None = None,
    ) -> None:
        """
        Save metadata to the entity sidecar.

        Parameters
        ----------
        entity_root_path : Path | str
            Function argument.
        updates : Mapping[str, Any]
            Function argument.
        target_image_name : str | None
            Function argument.
        image_names : Sequence[str] | None
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        if not updates:
            return

        del target_image_name, image_names
        entity_root = Path(entity_root_path)
        payload = {
            str(k): self._to_json_compatible(v) for k, v in dict(updates).items()
        }

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
        """
        Build write spec with OME transforms and extra attributes.

        Parameters
        ----------
        dtype : str | None
            Function argument.
        stage_zyx_um : Sequence[float] | None
            Function argument.
        extra_attributes : Mapping[str, Any] | None
            Function argument.

        Returns
        -------
        dict[str, Any]
            Function result.
        """

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

    def _update_image_translation_transform(
        self,
        image_root: Path,
        stage_zyx_um: Sequence[float],
    ) -> None:
        """
        Update an existing OME-Zarr image translation transform.

        Parameters
        ----------
        image_root : Path
            Image root path, without or with the ``.ome.zarr`` suffix.
        stage_zyx_um : Sequence[float]
            Physical Z, Y, X translation to write into the image metadata.

        Returns
        -------
        None
            The image metadata is updated in place when it already exists.
        """

        image_path = self._image_store_path(image_root)
        metadata_path = image_path / Path("zarr.json")
        if not metadata_path.exists():
            return

        metadata = self._load_from_json(metadata_path)
        if not isinstance(metadata, dict):
            return

        transforms = (
            metadata.get("attributes", {})
            .get("ome", {})
            .get("multiscales", [{}])[0]
            .get("datasets", [{}])[0]
            .get("coordinateTransformations", [])
        )
        for transform in transforms:
            if transform.get("type") == "translation":
                transform["translation"] = [float(v) for v in stage_zyx_um]
                self._save_to_json(metadata, metadata_path)
                return

    def _resolve_original_tile_position_zyx_um(
        self,
        tile_id: str,
        round_id: str | None = None,
        bit_id: str | None = None,
    ) -> list[float] | None:
        """
        Resolve original tile stage position used for OME translation.

        Parameters
        ----------
        tile_id : str
            Function argument.
        round_id : str | None
            Function argument.
        bit_id : str | None
            Function argument.

        Returns
        -------
        list[float] | None
            Function result.
        """

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

    def _resolve_reference_tile_position_zyx_um(
        self, tile_id: str
    ) -> list[float] | None:
        """
        Resolve the first-round reference stage position for registered outputs.

        Parameters
        ----------
        tile_id : str
            Tile identifier.

        Returns
        -------
        list[float] or None
            First fiducial round stage position in Z, Y, X microns.
        """

        if not getattr(self, "_round_ids", None):
            return None
        return self._resolve_original_tile_position_zyx_um(
            tile_id=tile_id, round_id=self._round_ids[0]
        )

    def _validate_core_image_shape(
        self,
        entity_root_path: Path | str,
        image_name: str,
        image: ArrayLike,
    ) -> None:
        """
        Enforce corrected/registered/feature-predictor image shape consistency.

        Parameters
        ----------
        entity_root_path : Path | str
            Function argument.
        image_name : str
            Function argument.
        image : ArrayLike
            Function argument.

        Returns
        -------
        None
            Function result.
        """

        entity_root = Path(entity_root_path)
        shape = tuple(int(v) for v in np.asarray(image).shape)
        required_names = {
            "corrected_data",
            "registered_decon_data",
            "decon_data",
            f"{self.feature_predictor_folder_name}_data",
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
                    "corrected_data, decon/registered data, and "
                    "feature_predictor_data must match."
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

    @staticmethod
    def _save_to_csv_gz(df: pd.DataFrame, csv_gz_path: Path | str) -> None:
        """Save dataframe to gzipped CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        csv_gz_path : Union[Path, str]
            Path to gzipped CSV file.
        """

        df.to_csv(csv_gz_path, index=False, compression="gzip")

    def _parse_datastore(self, validate: bool = True) -> None:
        """
        Parse datastore to discover available components.

        Parameters
        ----------
        validate : bool
            Function argument.

        Returns
        -------
        None
            Function result.
        """

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
                else:
                    setattr(self, "_" + key, attributes[key])

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
                psf_root_path = self._calibrations_zarr_path / Path("psf_data")
                try:
                    if psf_root_path.exists():

                        def psf_sort_key(path: Path) -> int:
                            return int(path.name[len("psf_") : len("psf_") + 3])

                        psf_dirs = sorted(
                            [
                                entry
                                for entry in psf_root_path.iterdir()
                                if entry.is_dir()
                                and re.fullmatch(r"psf_\d{3}\.ome\.zarr", entry.name)
                            ],
                            key=psf_sort_key,
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
        if self._datastore_state["Corrected"] and validate:
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
        if self._datastore_state["LocalRegistered"] and validate:
            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
                if round_id != self._round_ids[0]:
                    attributes = self._load_entity_attributes(entity_root)

                    keys_to_check = ["local_round_transform_zyx_um"]

                    for key in keys_to_check:
                        if key not in attributes.keys():
                            raise KeyError(
                                f"{round_id, tile_id} local round transform missing"
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
                current_local_zarr_path = str(entity_root / Path("decon_data"))

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
                    print(tile_id, bit_id)
                    print("Readout decon data missing.")

                current_local_zarr_path = str(
                    entity_root / Path(f"{self.feature_predictor_folder_name}_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except (OSError, ZarrError):
                    print(tile_id, bit_id)
                    print("feature_predictor prediction missing.")
                corrected_shape = self._image_shape(
                    entity_root / Path("corrected_data")
                )
                registered_shape = self._image_shape(entity_root / Path("decon_data"))
                feature_shape = self._image_shape(
                    entity_root / Path(f"{self.feature_predictor_folder_name}_data")
                )
                shapes = [
                    shape
                    for shape in (corrected_shape, registered_shape, feature_shape)
                    if shape is not None
                ]
                if len(shapes) > 1 and any(shape != shapes[0] for shape in shapes[1:]):
                    raise ValueError(
                        f"{tile_id} {bit_id} corrected/decon/feature image shapes differ: "
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
        if self._datastore_state["GlobalRegistered"] and validate:
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
        if self._datastore_state["Fused"] and validate:
            fused_image_path = self._fused_root_path / Path(
                f"fused_{self.fiducial_folder_name}_zyx"
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
        if self._datastore_state["SegmentedCells"] and validate:
            current_local_zarr_path = str(
                self._segmentation_root_path
                / Path("cellpose")
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
        if self._datastore_state["DecodedSpots"] and validate:
            for tile_id in self._tile_ids:
                decoded_path = self._decoded_root_path / Path(
                    tile_id + "_decoded_features.parquet"
                )

                if not (decoded_path.exists()):
                    raise FileNotFoundError(tile_id + " decoded spots missing.")

        # check and validate filtered decoded spots
        if self._datastore_state["FilteredSpots"] and validate:
            filtered_path = self._decoded_root_path / Path(
                "all_tiles_filtered_decoded_features.parquet"
            )

            if not (filtered_path.exists()):
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
        """Load readout bits linked to fiducial round for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        round : Union[int, str]
            Round index or round id.

        Returns
        -------
        bit_linker : Optional[Sequence[int]]
            Readout bits linked to fiducial round for one tile.
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
        """Save readout bits linked to fiducial round for one tile.

        Parameters
        ----------
        bit_linker : Sequence[int]
            Readout bits linked to fiducial round for one tile.
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
        """Load fiducial round linked to readout bit for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        bit : Union[int, str]
            Bit index or bit id.

        Returns
        -------
        round_linker : Optional[Sequence[int]]
            Fiducial round linked to readout bit for one tile.
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
        """Save fiducial round linker attribute to readout bit for one tile.

        Parameters
        ----------
        round_linker : int
            Fiducial round linked to readout bit for one tile.
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
            stage_zyx_um = np.asarray(stage_zyx_um, dtype=np.float32)
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "stage_zyx_um": stage_zyx_um.tolist(),
                    "affine_zyx_px": np.asarray(
                        affine_zyx_px, dtype=np.float32
                    ).tolist(),
                },
                target_image_name="corrected_data",
            )
            self._update_image_translation_transform(
                entity_root / Path("corrected_data"), stage_zyx_um
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
        """Load wavelengths for fiducial OR readout bit for one tile.

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
            Wavelengths for fiducial OR readout bit for one tile.
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
        """Save wavelengths for fiducial OR readout bit for one tile.

        Parameters
        ----------
        wavelengths_um : tuple[float, float]
            Wavelengths for fiducial OR readout bit for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        round : Optional[Union[int, str]]
            Round index or round id.
        bit : Optional[Union[int, str]]
            Bit index or bit id.

        Returns
        -------
        wavelengths_um : Optional[tuple[float, float]]
            Wavelengths for fiducial OR readout bit for one tile.
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
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if round < 0:
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
        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            attributes = self._load_entity_attributes(entity_root)
            return np.asarray(
                attributes["local_round_transform_zyx_um"], dtype=np.float32
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(tile_id, round_id)
            print("Local round transform mapping back to first round not found.")
            return None

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
            The transform is stored in the entity attributes for the requested
            fiducial tile and round.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if round < 0:
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
        try:
            entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates={
                    "local_round_transform_zyx_um": np.asarray(
                        transform_zyx_um, dtype=np.float32
                    ).tolist()
                },
                target_image_name="registered_decon_data",
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            print("Error writing local round transform attribute.")
            return None

    def load_coord_of_xform_px(
        self,
        tile: int | str | None,
        round: int | str | None,
        return_future: bool | None = True,
    ) -> tuple[ArrayLike, ArrayLike] | None:
        """Local fiducial optical flow matrix for one round and tile.

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
            Local fiducial optical flow matrix for one round and tile.
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
        """Save fiducial optical flow matrix for one round and tile.

        Parameters
        ----------
        of_xform_px : ArrayLike
            Local fiducial optical flow matrix for one round and tile.
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

    def load_local_sofima_flow_field(
        self,
        *,
        tile: int | str,
        round: int | str,
        return_future: bool | None = True,
    ) -> tuple[ArrayLike, dict] | None:
        """
        Load the SOFIMA flow field for one local fiducial round.

        Parameters
        ----------
        tile : int or str
            Tile index or tile identifier.
        round : int or str
            Moving fiducial round index or identifier.
        return_future : bool, default=True
            If True, return the lazy array object used by the datastore backend.

        Returns
        -------
        tuple[ArrayLike, dict] or None
            SOFIMA flow field and metadata attributes. The map channels are X,
            Y, Z and spatial axes are Z, Y, X. ``map_stride_zyx_px`` is stored
            in Z, Y, X order. ``map_box_start_xyz_px`` is stored in X, Y, Z
            order and gives the reference-grid coordinate of the first flow
            sample. For fields produced by SOFIMA this is the patch center
            coordinate, not the image corner.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if round < 0:
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

        image_name = "local_sofima_flow_field"
        entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
        current_local_zarr_path = entity_root / Path(image_name)
        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("SOFIMA flow field not found.")
            return None

        try:
            spec = self._build_image_write_spec(dtype="<f4")
            sofima_flow_field = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            attributes = self._load_entity_attributes(
                entity_root,
                image_names=(image_name,),
            )
            return sofima_flow_field, attributes
        except (OSError, ZarrError, KeyError) as e:
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
    ) -> None:
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
        return_future : bool, default=False
            If True, return the asynchronous datastore write object.

        Returns
        -------
        None
            The flow field and attributes are written to the datastore.
        """

        if isinstance(tile, int):
            if tile < 0 or tile > self._num_tiles:
                print("Set tile index >=0 and <=" + str(self._num_tiles))
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
            if round < 0:
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
            reference_round_id = self._round_ids[reference_round]
        else:
            reference_round_id = str(reference_round)

        image_name = "local_sofima_flow_field"
        entity_root = self._fiducial_root_path / Path(tile_id) / Path(round_id)
        current_local_zarr_path = entity_root / Path(image_name)
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
            "map_stride_zyx_px": np.asarray(
                map_stride_zyx_px, dtype=np.float32
            ).tolist(),
            "map_box_start_xyz_px": np.asarray(
                map_box_start_xyz_px, dtype=np.float32
            ).tolist(),
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
            spec = self._build_image_write_spec(
                dtype="<f4",
                extra_attributes=attributes,
            )
            self._save_to_zarr_array(
                np.asarray(sofima_flow_field_xyz_px, dtype=np.float32),
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            self._save_entity_attributes(
                entity_root_path=entity_root,
                updates=attributes,
                target_image_name=image_name,
                image_names=(image_name,),
            )
        except (OSError, TimeoutError):
            print("Error saving SOFIMA flow field.")
            return None

    def load_local_registered_image(
        self,
        tile: int | str,
        round: int | str | None = None,
        bit: int | str | None = None,
        return_future: bool | None = True,
    ) -> ArrayLike | None:
        """Load a fiducial registered image or an unwarped readout image.

        Fiducial rounds are loaded from ``registered_decon_data`` after local
        registration. Readout bits are loaded from ``decon_data`` in their
        native, unwarped tile frame; pixel decoding applies fiducial, SOFIMA,
        and chromatic transforms at load time.

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
        ArrayLike or None
            Fiducial registered image or unwarped readout image.
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
                / Path("decon_data")
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
            image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            return image
        except (OSError, ZarrError) as e:
            print(e)
            print("Error loading local deconvolved image.")
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
        """Save a fiducial registered image or an unwarped readout image.

        Fiducial rounds are saved under ``registered_decon_data`` after local
        registration. Readout bits are saved under ``decon_data`` in their
        native, unwarped tile frame.

        Parameters
        ----------
        registered_image : ArrayLike
            Image to save.
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
            current_local_zarr_path = entity_root / Path("decon_data")
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
            stage_position = self._resolve_reference_tile_position_zyx_um(tile_id)

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name="decon_data" if bit is not None else "registered_decon_data",
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
                target_image_name=(
                    "decon_data" if bit is not None else "registered_decon_data"
                ),
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
        feature_predictor_image : Optional[ArrayLike]
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
            / Path(f"{self.feature_predictor_folder_name}_data")
        )

        image_path = self._image_store_path(current_local_zarr_path)
        if not image_path.exists():
            print("feature_predictor prediction image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<f4"
            feature_predictor_image = self._load_from_zarr_array(
                self._get_kvstore_key(image_path),
                spec,
                return_future,
            )
            return feature_predictor_image
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
                f"{self.feature_predictor_folder_name}_data"
            )

        try:
            self._validate_core_image_shape(
                entity_root_path=entity_root,
                image_name=f"{self.feature_predictor_folder_name}_data",
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
                target_image_name=f"{self.feature_predictor_folder_name}_data",
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

    def load_global_fiducial_image(
        self,
        return_future: bool | None = True,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike] | None:
        """Load downsampled, fused fiducial image.

        Parameters
        ----------
        return_future : Optional[bool]
            Return future array.

        Returns
        -------
        fused_image : Optional[ArrayLike]
            Downsampled, fused fiducial image.
        affine_zyx_um : Optional[ArrayLike]
            Global affine registration transform for fused image.
        origin_zyx_um : Optional[ArrayLike]
            Global origin registration transform for fused image.
        spacing_zyx_um : Optional[ArrayLike]
            Global spacing registration transform for fused image.
        """

        current_local_zarr_path = self._fused_root_path / Path(
            f"fused_{self.fiducial_folder_name}_zyx"
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

    def save_global_fiducial_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        fusion_type: str = "fiducial",
        return_future: bool | None = False,
    ) -> None:
        """Save downsampled, fused fiducial image.

        Parameters
        ----------
        fused_image : ArrayLike
            Downsampled, fused fiducial image.
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
            filename = f"fused_{self.fiducial_folder_name}_zyx"
        else:
            filename = "fused_all_channels_zyx"
        current_local_zarr_path = self._fused_root_path / Path(filename)

        metadata_attrs = {
            "affine_zyx_um": np.asarray(affine_zyx_um, dtype=np.float32).tolist(),
            "origin_zyx_um": np.asarray(origin_zyx_um, dtype=np.float32).tolist(),
            "spacing_zyx_um": np.asarray(spacing_zyx_um, dtype=np.float32).tolist(),
        }
        fused_array = np.asarray(fused_image)
        try:
            spec = self._build_image_write_spec(
                dtype="<u2",
                extra_attributes=metadata_attrs,
            )
            spec["metadata"]["chunks"] = self._fused_image_chunks(fused_array)
            self._save_to_zarr_array(
                fused_array.astype(np.uint16),
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
        decode_run_key: str | None = None,
    ) -> pd.DataFrame | None:
        """Load decoded spots and features for one tile.

        Parameters
        ----------
        tile : Union[int, str]
            Tile index or tile id.
        decode_run_key : str or None, default None
            Optional decoded-output run key.

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

        current_tile_features_path = self._decoded_run_root(decode_run_key) / Path(
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
        decode_run_key: str | None = None,
    ) -> None:
        """Save decoded spots and features for one tile.

        Parameters
        ----------
        features_df : pd.DataFrame
            Decoded spots and features for one tile.
        tile : Union[int, str]
            Tile index or tile id.
        decode_run_key : str or None, default None
            Optional decoded-output run key.
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

        decoded_root_path = self._decoded_run_root(decode_run_key)
        decoded_root_path.mkdir(parents=True, exist_ok=True)
        current_tile_features_path = decoded_root_path / Path(
            tile_id + "_decoded_features.parquet"
        )

        self._save_to_parquet(features_df, current_tile_features_path)

    def load_global_filtered_decoded_spots(
        self,
        decode_run_key: str | None = None,
    ) -> pd.DataFrame | None:
        """Load all decoded and filtered spots.

        Parameters
        ----------
        decode_run_key : str or None, default None
            Optional decoded-output run key.

        Returns
        -------
        all_tiles_filtered : Optional[pd.DataFrame]
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
