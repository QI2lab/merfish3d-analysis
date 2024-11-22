"""
qi2labDataStore: Interface to qi2lab MERFISH datastore

Shepherd 2024/07 - initial commit
"""

from typing import Union, Optional, Sequence, Collection
from pathlib import Path
from numpy.typing import ArrayLike
import tensorstore as ts
import pandas as pd
import numpy as np
import json
from itertools import product


class qi2labDataStore:
    """API to qi2lab MERFISH store.

    Parameters
    ----------
    datastore_path : Union[str, Path]
        Path to qi2lab MERFISH store

    """

    def __init__(self, datastore_path: Union[str, Path]):
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
    def datastore_state(self) -> Optional[dict]:
        """Datastore state."""

        return getattr(self, "_datastore_state", None)

    @datastore_state.setter
    def datastore_state(self, value: dict):
        if not hasattr(self, "_datastore_state") or self._datastore_state is None:
            self._datastore_state = value
        else:
            self._datastore_state.update(value)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @property
    def microscope_type(self) -> Optional[str]:
        """Microscope type."""

        return getattr(self, "_microscope_type", None)

    @microscope_type.setter
    def microscope_type(self, value: str):
        self._microscope_type = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["microscope_type"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def camera_model(self) -> Optional[str]:
        """Camera model."""

        return getattr(self, "_camera_model", None)

    @camera_model.setter
    def camera_model(self, value: str):
        self._camera_model = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["camera_model"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def num_rounds(self) -> Optional[int]:
        """Number of rounds."""

        return getattr(self, "_num_rounds", None)

    @num_rounds.setter
    def num_rounds(self, value: int):
        self._num_rounds = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["num_rounds"] = value
        self._save_to_json(calib_zattrs, zattrs_path)
        
    @property
    def num_bits(self) -> int:
        return getattr(self, "_num_bits", None)

    @property
    def num_tiles(self) -> Optional[int]:
        """Number of tiles."""

        return getattr(self, "_num_tiles", None)

    @num_tiles.setter
    def num_tiles(self, value: int):
        self._num_tiles = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["num_tiles"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

        self._tile_ids = []
        for tile_idx in range(value):
            self._tile_ids.append("tile" + str(tile_idx).zfill(4))

    @property
    def channels_in_data(self) -> Optional[Collection[int]]:
        """Channel names."""

        return getattr(self, "_channels_in_data", None)

    @channels_in_data.setter
    def channels_in_data(self, value: Collection[int]):
        self._channels_in_data = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["channels_in_data"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def tile_overlap(self) -> Optional[float]:
        """XY tile overlap."""

        return getattr(self, "_tile_overlap", None)

    @tile_overlap.setter
    def tile_overlap(self, value: float):
        self._tile_overlap = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["tile_overlap"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def binning(self) -> Optional[int]:
        """Camera binning."""

        return getattr(self, "_binning", None)

    @binning.setter
    def binning(self, value: int):
        self._binning = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["binning"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def e_per_ADU(self) -> Optional[float]:
        """Electrons per camera ADU."""

        return getattr(self, "_e_per_ADU", None)

    @e_per_ADU.setter
    def e_per_ADU(self, value: float):
        self._e_per_ADU = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["e_per_ADU"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def na(self) -> Optional[float]:
        """Detection objective numerical aperture (NA)."""

        return getattr(self, "_na", None)

    @na.setter
    def na(self, value: float):
        self._na = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["na"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def ri(self) -> Optional[float]:
        """Detection objective refractive index (RI)."""

        return getattr(self, "_ri", None)

    @ri.setter
    def ri(self, value: float):
        self._ri = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["ri"] = value
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def noise_map(self) -> Optional[ArrayLike]:
        """Camera noise image."""

        return getattr(self, "_noise_map", None)

    @noise_map.setter
    def noise_map(self, value: ArrayLike):
        self._noise_map = value
        current_local_zarr_path = str(self._calibrations_zarr_path / Path("noise_map"))

        try:
            self._save_to_zarr_array(
                value,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future=False,
            )
        except Exception as e:
            print(e)
            print(r"Could not access calibrations.zarr/noise_map")

    @property
    def channel_shading_maps(self) -> Optional[ArrayLike]:
        """Channel shaiding images."""

        return getattr(self, "_shading_maps", None)

    @channel_shading_maps.setter
    def channel_shading_maps(self, value: ArrayLike):
        self._shading_maps = value
        current_local_zarr_path = str(
            self._calibrations_zarr_path / Path("shading_maps")
        )

        try:
            self._save_to_zarr_array(
                value,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future=False,
            )
        except Exception:
            print(r"Could not access calibrations.zarr/shading_maps")

    @property
    def channel_psfs(self) -> Optional[ArrayLike]:
        """Channel point spread functions (PSF)."""
        return getattr(self, "_psfs", None)

    @channel_psfs.setter
    def channel_psfs(self, value: ArrayLike):
        self._psfs = value
        current_local_zarr_path = str(self._calibrations_zarr_path / Path("psf_data"))

        try:
            self._save_to_zarr_array(
                value,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future=False,
            )
        except Exception:
            print(r"Could not access calibrations.zarr/psf_data")

    @property
    def experiment_order(self) -> Optional[pd.DataFrame]:
        """Round and bit order."""

        return getattr(self, "_experiment_order", None)

    @experiment_order.setter
    def experiment_order(self, value: Union[ArrayLike, pd.DataFrame]):
        if isinstance(value, pd.DataFrame):
            self._experiment_order = value
        else:
            channel_list = []
            for idx in range(len(self._channels_in_data)):
                channel_list.append(str(self._channels_in_data[idx]))
            self._experiment_order = pd.DataFrame(
                value, columns=channel_list, dtype="int64"
            )

        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["exp_order"] = self._experiment_order.values.tolist()
        self._save_to_json(calib_zattrs, zattrs_path)

        self._num_rounds = int(value[-1, 0])
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["num_round"] = self._num_rounds
        self._save_to_json(calib_zattrs, zattrs_path)

        self._num_bits = int(np.max(value[:, 1:]))
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["num_bits"] = self._num_bits
        self._save_to_json(calib_zattrs, zattrs_path)

        self._round_ids = []
        for round_idx in range(self._num_rounds):
            self._round_ids.append("round" + str(round_idx + 1).zfill(3))

        self._bit_ids = []
        for bit_idx in range(self._num_bits):
            self._bit_ids.append("bit" + str(bit_idx + 1).zfill(3))

    @property
    def codebook(self) -> Optional[pd.DataFrame]:
        """Codebook."""

        data = getattr(self, "_codebook", None)

        if data is None:
            return None
        num_columns = len(data[0]) if data else 0
        columns = ["gene_id"] + [f"bit{i:02d}" for i in range(1, num_columns)]

        return pd.DataFrame(data, columns=columns)

    @codebook.setter
    def codebook(self, value: pd.DataFrame):
        self._codebook = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["codebook"] = self._codebook.values.tolist()
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def voxel_size_zyx_um(self) -> Optional[ArrayLike]:
        """Voxel size, zyx order (microns)."""

        return getattr(self, "_voxel_size_zyx_um", None)

    @voxel_size_zyx_um.setter
    def voxel_size_zyx_um(self, value: ArrayLike):
        self._voxel_size_zyx_um = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["voxel_size_zyx_um"] = value
        self._save_to_json(calib_zattrs, zattrs_path)
        
    @property
    def baysor_path(self) -> Union[Path,str]:
        """Baysor path"""
        return getattr(self,"_baysor_path",None)
    
    @baysor_path.setter
    def baysor_path(self, value: Union[Path,str]):
        self._baysor_path = Path(value)
        self._datastore_state["BaysorPath"] = str(self._baysor_path)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @property
    def baysor_options(self) -> Union[Path,str]:
        """Baysor options"""
        return getattr(self,"_baysor_options",None)
    
    @baysor_options.setter
    def baysor_options(self, value: Union[Path,str]):
        self._baysor_options = Path(value)
        self._datastore_state["BaysorOptions"] = str(self._baysor_options)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @property
    def julia_threads(self) -> int:
        """Julia thread number"""
        return getattr(self,"_julia_threads",None)
    
    @julia_threads.setter
    def julia_threads(self, value: int):
        self._julia_threads = value
        self._datastore_state["JuliaThreads"] = str(self._julia_threads)
        self._save_to_json(self._datastore_state, self._datastore_state_json_path)
        
    @property
    def global_normalization_vector(self) -> Optional[ArrayLike]:
        """Global normalization vector."""

        value = getattr(self, "_global_normalization_vector", None)
        if value is None:
            zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
            calib_zattrs = self._load_from_json(zattrs_path)

            try:
                value = np.asarray(
                    calib_zattrs["global_normalization_vector"], dtype=np.float32
                )
                return value
            except Exception:
                print("Global normalization vector not calculated.")
                return None
        else:
            return value

    @global_normalization_vector.setter
    def global_normalization_vector(self, value: ArrayLike):
        self._global_normalization_vector = np.asarray(value, dtype=np.float32)
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["global_normalization_vector"] = (
            self._global_normalization_vector.tolist()
        )
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def global_background_vector(self) -> Optional[ArrayLike]:
        """Global background vector."""

        value = getattr(self, "_global_background_vector", None)
        if value is None:
            zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
            calib_zattrs = self._load_from_json(zattrs_path)
            try:
                value = np.asarray(
                    calib_zattrs["global_background_vector"], dtype=np.float32
                )
                return value
            except Exception:
                print("Global background vector not calculated.")
                return None
        else:
            return value

    @global_background_vector.setter
    def global_background_vector(self, value: ArrayLike):
        self._global_background_vector = np.asarray(value, dtype=np.float32)
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["global_background_vector"] = (
            self._global_background_vector.tolist()
        )
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def iterative_normalization_vector(self) -> Optional[ArrayLike]:
        """Iterative normalization vector."""

        value = getattr(self, "_iterative_normalization_vector", None)
        if value is None:
            zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
            calib_zattrs = self._load_from_json(zattrs_path)
            try:
                value = np.asarray(
                    calib_zattrs["iterative_normalization_vector"], dtype=np.float32
                )
            except Exception:
                value = None

            if value is None:
                print("Iterative normalization vector not calculated.")
                return None

            return value
        else:
            return value

    @iterative_normalization_vector.setter
    def iterative_normalization_vector(self, value: ArrayLike):
        self._iterative_normalization_vector = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["iterative_normalization_vector"] = (
            self._iterative_normalization_vector.tolist()
        )
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def iterative_background_vector(self) -> Optional[ArrayLike]:
        """Iterative background vector."""

        value = getattr(self, "_iterative_background_vector", None)
        if value is None:
            zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
            calib_zattrs = self._load_from_json(zattrs_path)
            try:
                value = np.asarray(
                    calib_zattrs["iterative_background_vector"], dtype=np.float32
                )
            except Exception:
                value = None
            if value is None:
                print("Iterative background vector not calculated.")
                return None

            return value
        else:
            return value

    @iterative_background_vector.setter
    def iterative_background_vector(self, value: ArrayLike):
        self._iterative_background_vector = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["iterative_background_vector"] = (
            self._iterative_background_vector.tolist()
        )
        self._save_to_json(calib_zattrs, zattrs_path)

    @property
    def tile_ids(self) -> Optional[Collection[str]]:
        """Tile IDs."""

        return getattr(self, "_tile_ids", None)

    @property
    def round_ids(self) -> Optional[Collection[str]]:
        """Round IDs."""

        return getattr(self, "_round_ids", None)

    @property
    def bit_ids(self) -> Optional[Collection[str]]:
        """Bit IDs."""

        return getattr(self, "_bit_ids", None)

    def _init_datastore(self):
        """Initialize datastore."""

        self._datastore_path.mkdir(parents=True)
        self._calibrations_zarr_path = self._datastore_path / Path(r"calibrations.zarr")
        self._calibrations_zarr_path.mkdir()
        calibrations_zattrs_path = self._calibrations_zarr_path / Path(r".zattrs")
        empty_zattrs = {}
        self._save_to_json(empty_zattrs, calibrations_zattrs_path)
        self._polyDT_root_path = self._datastore_path / Path(r"polyDT")
        self._polyDT_root_path.mkdir()
        self._readouts_root_path = self._datastore_path / Path(r"readouts")
        self._readouts_root_path.mkdir()
        self._ufish_localizations_root_path = self._datastore_path / Path(
            r"ufish_localizations"
        )
        self._ufish_localizations_root_path.mkdir()
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
            "Version": 0.3,
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
            "JuliaThreads": str(self._julia_threads)
        }

        self._save_to_json(self._datastore_state, self._datastore_state_json_path)

    @staticmethod
    def _get_kvstore_key(path: Union[Path, str]) -> dict:
        """Convert datastore location to tensorstore kvstore key"""
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
    def _load_from_json(dictionary_path: Union[Path, str]) -> dict:
        """Load json as dictionary."""

        try:
            with open(dictionary_path, "r") as f:
                dictionary = json.load(f)
        except Exception:
            dictionary = {}
        return dictionary

    @staticmethod
    def _save_to_json(dictionary: dict, dictionary_path: Union[Path, str]):
        """Save dictionary to json."""

        with open(dictionary_path, "w") as file:
            json.dump(dictionary, file, indent=4)

    @staticmethod
    def _load_from_microjson(dictionary_path: Union[Path, str]) -> dict:
        """Load cell outlines outlines microjson as dictionary"""

        try:
            with open(dictionary_path, "r") as f:
                data = json.load(f)
                outlines = {}
                for feature in data["features"]:
                    cell_id = feature["properties"]["cell_id"]
                    coordinates = feature["geometry"]["coordinates"][0]
                    outlines[cell_id] = np.array(coordinates)
        except Exception:
            outlines = {}
        return outlines

    @staticmethod
    def _check_for_zarr_array(kvstore: Union[Path, str], spec: dict):
        """Check if zarr existing using tensortore."""

        current_zarr = ts.open(
            {
                **spec,
                "kvstore": kvstore,
            }
        ).result()

        del current_zarr

    @staticmethod
    def _load_from_zarr_array(
        kvstore: dict, spec: dict, return_future=True
    ) -> ArrayLike:
        """Return tensorstore array from zarr

        Defaults to returning future result.
        """

        current_zarr = ts.open(
            {
                **spec,
                "kvstore": kvstore,
            }
        ).result()

        read_future = current_zarr.read()

        if return_future:
            return read_future
        else:
            return read_future.result()

    @staticmethod
    def _save_to_zarr_array(
        array: ArrayLike,
        kvstore: dict,
        spec: dict,
        return_future: Optional[bool] = False,
    ) -> Optional[ArrayLike]:
        """Save array to zarr using tensorstore.

        Defaults to returning future result.
        """
        if str(array.dtype) == "uint8":
            array_dtype = "<u1"
        elif str(array.dtype) == "uint16":
            array_dtype = "<u2"
        elif str(array.dtype) == "float16":
            array_dtype = "<f2"
        elif str(array.dtype) == "float32":
            array_dtype = "<f4"
        else:
            print("Unsupported data type: " + str(array.dtype))
            return None

        spec["metadata"]["shape"] = array.shape
        if len(array.shape) == 2:
            spec["metadata"]["chunks"] = [array.shape[0], array.shape[1]]
        elif len(array.shape) == 3:
            spec["metadata"]["chunks"] = [1, array.shape[1], array.shape[2]]
        elif len(array.shape) == 4:
            spec["metadata"]["chunks"] = [1, 1, array.shape[1], array.shape[2]]
        spec["metadata"]["dtype"] = array_dtype

        try:
            current_zarr = ts.open(
                {
                    **spec,
                    "kvstore": kvstore,
                }
            ).result()

            write_future = current_zarr.write(array)

            if return_future:
                return write_future
            else:
                write_future.result()
                return None
        except Exception as e:
            print(e)
            print("Error writing zarr array.")

    @staticmethod
    def _load_from_parquet(parquet_path: Union[Path, str]) -> pd.DataFrame:
        """Load dataframe from parquet."""

        return pd.read_parquet(parquet_path)

    @staticmethod
    def _save_to_parquet(df: pd.DataFrame, parquet_path: Union[Path, str]):
        """Save dataframe to parquet."""

        df.to_parquet(parquet_path)

    def _parse_datastore(self):
        """Parse datastore to discover available components."""

        # directory structure as defined by qi2lab spec
        self._calibrations_zarr_path = self._datastore_path / Path(r"calibrations.zarr")
        self._polyDT_root_path = self._datastore_path / Path(r"polyDT")
        self._readouts_root_path = self._datastore_path / Path(r"readouts")
        self._ufish_localizations_root_path = self._datastore_path / Path(
            r"ufish_localizations"
        )
        self._decoded_root_path = self._datastore_path / Path(r"decoded")
        self._fused_root_path = self._datastore_path / Path(r"fused")
        self._segmentation_root_path = self._datastore_path / Path(r"segmentation")
        self._mtx_output_root_path = self._datastore_path / Path(r"mtx_output")
        self._datastore_state_json_path = self._datastore_path / Path(
            r"datastore_state.json"
        )

        # read in .json in root directory that indicates what steps have been run
        with open(self._datastore_state_json_path, "r") as json_file:
            self._datastore_state = json.load(json_file)

        # validate calibrations.zarr
        if self._datastore_state["Calibrations"]:
            if not (self._calibrations_zarr_path.exists()):
                print("Calibration data error.")
            try:
                zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
                attributes = self._load_from_json(zattrs_path)
            except Exception:
                print("Calibration attributes not found")

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
                "num_bits"
            ]
            if self._datastore_state["Version"] == 0.3:
                keys_to_check.append("microscope_type")
                keys_to_check.append("camera_model")
                keys_to_check.append("voxel_size_zyx_um")
            for key in keys_to_check:
                if key not in attributes.keys():
                    raise Exception("Calibration attributes incomplete")
                else:
                    setattr(self, "_" + key, attributes[key])

            current_local_zarr_path = str(
                self._calibrations_zarr_path / Path("psf_data")
            )

            try:
                self._psfs = (
                    self._load_from_zarr_array(
                        kvstore=self._get_kvstore_key(current_local_zarr_path),
                        spec=self._zarrv2_spec.copy(),
                    )
                ).result()
            except Exception:
                print("Calibration psfs missing.")

            del current_local_zarr_path

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

        # validate polyDT and readout bits data
        if self._datastore_state["Corrected"]:
            if not (self._polyDT_root_path.exists()):
                raise Exception("PolyDT directory not initialized")
            else:
                polyDT_tile_ids = sorted(
                    [
                        entry.name
                        for entry in self._polyDT_root_path.iterdir()
                        if entry.is_dir()
                    ],
                    key=lambda x: int(x.split("tile")[1].split(".zarr")[0]),
                )
                current_tile_dir_path = self._polyDT_root_path / Path(
                    polyDT_tile_ids[0]
                )
                self._round_ids = sorted(
                    [
                        entry.name.split(".")[0]
                        for entry in current_tile_dir_path.iterdir()
                        if entry.is_dir()
                    ],
                    key=lambda x: int(x.split("round")[1].split(".zarr")[0]),
                )
            if not (self._readouts_root_path.exists()):
                raise Exception("Readout directory not initialized")
            else:
                readout_tile_ids = sorted(
                    [
                        entry.name
                        for entry in self._readouts_root_path.iterdir()
                        if entry.is_dir()
                    ],
                    key=lambda x: int(x.split("tile")[1].split(".zarr")[0]),
                )
                current_tile_dir_path = self._readouts_root_path / Path(
                    readout_tile_ids[0]
                )
                self._bit_ids = sorted(
                    [
                        entry.name.split(".")[0]
                        for entry in current_tile_dir_path.iterdir()
                        if entry.is_dir()
                    ],
                    key=lambda x: int(x.split("bit")[1].split(".zarr")[0]),
                )
            assert (
                polyDT_tile_ids == readout_tile_ids
            ), "polyDT and readout tile ids do not match. Conversion error."
            self._tile_ids = polyDT_tile_ids.copy()
            del polyDT_tile_ids, readout_tile_ids

            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                try:
                    zattrs_path = str(
                        self._polyDT_root_path
                        / Path(tile_id)
                        / Path(round_id + ".zarr")
                        / Path(".zattrs")
                    )
                    attributes = self._load_from_json(zattrs_path)
                except Exception:
                    print("polyDT tile attributes not found")

                keys_to_check = [
                    "stage_zyx_um",
                    "excitation_um",
                    "emission_um",
                    "bit_linker",
                    # "exposure_ms",
                    "psf_idx",
                ]

                for key in keys_to_check:
                    if key not in attributes.keys():
                        print(tile_id, round_id, key)
                        raise Exception("Corrected polyDT attributes incomplete")

                current_local_zarr_path = str(
                    self._polyDT_root_path
                    / Path(tile_id)
                    / Path(round_id + ".zarr")
                    / Path("corrected_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except Exception:
                    print(tile_id, round_id)
                    print("Corrected polyDT data missing.")

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                try:
                    zattrs_path = str(
                        self._readouts_root_path
                        / Path(tile_id)
                        / Path(bit_id + ".zarr")
                        / Path(".zattrs")
                    )
                    attributes = self._load_from_json(zattrs_path)
                except Exception:
                    print("Readout tile attributes not found")

                keys_to_check = [
                    "excitation_um",
                    "emission_um",
                    "round_linker",
                    # "exposure_ms",
                    "psf_idx",
                ]
                for key in keys_to_check:
                    if key not in attributes.keys():
                        raise Exception("Corrected readout attributes incomplete")

                current_local_zarr_path = str(
                    self._readouts_root_path
                    / Path(tile_id)
                    / Path(bit_id + ".zarr")
                    / Path("corrected_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except Exception:
                    print(tile_id, bit_id)
                    print("Corrected readout data missing.")

        # check and validate local registered data
        if self._datastore_state["LocalRegistered"]:
            for tile_id, round_id in product(self._tile_ids, self._round_ids):
                if round_id is not self._round_ids[0]:
                    try:
                        zattrs_path = str(
                            self._polyDT_root_path
                            / Path(tile_id)
                            / Path(round_id + ".zarr")
                            / Path(".zattrs")
                        )
                        with open(zattrs_path, "r") as f:
                            attributes = json.load(f)
                    except Exception:
                        print("polyDT tile attributes not found")

                    keys_to_check = ["rigid_xform_xyz_px"]

                    for key in keys_to_check:
                        if key not in attributes.keys():
                            raise Exception("Rigid registration missing")

                    current_local_zarr_path = str(
                        self._polyDT_root_path
                        / Path(tile_id)
                        / Path(round_id + ".zarr")
                        / Path("of_xform_px")
                    )

                    try:
                        self._check_for_zarr_array(
                            self._get_kvstore_key(current_local_zarr_path),
                            self._zarrv2_spec.copy(),
                        )
                    except Exception:
                        print(tile_id, round_id)
                        print("Optical flow registration data missing.")

                current_local_zarr_path = str(
                    self._polyDT_root_path
                    / Path(tile_id)
                    / Path(round_id + ".zarr")
                    / Path("registered_decon_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except Exception:
                    print(tile_id, round_id)
                    print("Registered polyDT data missing.")

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                current_local_zarr_path = str(
                    self._readouts_root_path
                    / Path(tile_id)
                    / Path(bit_id + ".zarr")
                    / Path("registered_decon_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except Exception:
                    print(tile_id, round_id)
                    print("Registered readout data missing.")

                current_local_zarr_path = str(
                    self._readouts_root_path
                    / Path(tile_id)
                    / Path(bit_id + ".zarr")
                    / Path("registered_ufish_data")
                )

                try:
                    self._check_for_zarr_array(
                        self._get_kvstore_key(current_local_zarr_path),
                        self._zarrv2_spec.copy(),
                    )
                except Exception:
                    print(tile_id, round_id)
                    print("Registered ufish prediction missing.")

            for tile_id, bit_id in product(self._tile_ids, self._bit_ids):
                current_ufish_path = (
                    self._ufish_localizations_root_path
                    / Path(tile_id)
                    / Path(bit_id + ".parquet")
                )
                if not (current_ufish_path.exists()):
                    raise Exception(
                        tile_id + " " + bit_id + " ufish localization missing"
                    )

        # check and validate global registered data
        if self._datastore_state["GlobalRegistered"]:
            for tile_id in self._tile_ids:
                try:
                    zattrs_path = str(
                        self._polyDT_root_path
                        / Path(tile_id)
                        / Path(self._round_ids[0] + ".zarr")
                        / Path(".zattrs")
                    )
                    with open(zattrs_path, "r") as f:
                        attributes = json.load(f)
                except Exception:
                    print("polyDT tile attributes not found")

                keys_to_check = ["affine_zyx_um", "origin_zyx_um", "spacing_zyx_um"]

                for key in keys_to_check:
                    if key not in attributes.keys():
                        raise Exception("Global registration missing")

        # check and validate fused
        if self._datastore_state["Fused"]:
            try:
                zattrs_path = str(
                    self._fused_root_path
                    / Path("fused.zarr")
                    / Path("fused_polyDT_iso_zyx")
                    / Path(".zattrs")
                )
                with open(zattrs_path, "r") as f:
                    attributes = json.load(f)
            except Exception:
                print("Fused image attributes not found")

            keys_to_check = ["affine_zyx_um", "origin_zyx_um", "spacing_zyx_um"]

            for key in keys_to_check:
                if key not in attributes.keys():
                    raise Exception("Fused image metadata missing")

            current_local_zarr_path = str(
                self._fused_root_path
                / Path("fused.zarr")
                / Path("fused_polyDT_iso_zyx")
            )

            try:
                self._check_for_zarr_array(
                    self._get_kvstore_key(current_local_zarr_path),
                    self._zarrv2_spec.copy(),
                )
            except Exception:
                print("Fused data missing.")

        # check and validate cellpose segmentation
        if self._datastore_state["SegmentedCells"]:
            current_local_zarr_path = str(
                self._segmentation_root_path
                / Path("cellpose")
                / Path("cellpose.zarr")
                / Path("masks_polyDT_iso_zyx")
            )

            try:
                self._check_for_zarr_array(
                    self._get_kvstore_key(current_local_zarr_path),
                    self._zarrv2_spec.copy(),
                )
            except Exception:
                print("Cellpose data missing.")

            cell_centroids_path = (
                self._segmentation_root_path
                / Path("cellpose")
                / Path("cell_centroids.parquet")
            )
            if not (cell_centroids_path.exists()):
                raise Exception("Cellpose cell centroids missing.")
            cell_outlines_path = (
                self._segmentation_root_path
                / Path("cellpose")
                / Path("cell_outlines.json")
            )
            if not (cell_outlines_path.exists()):
                raise Exception("Cellpose cell oultines missing.")

        # check and validate decoded spots
        if self._datastore_state["DecodedSpots"]:
            for tile_id in self._tile_ids:
                decoded_path = self._decoded_root_path / Path(
                    tile_id + "_decoded_features.parquet"
                )

                if not (decoded_path.exists()):
                    raise Exception(tile_id + " decoded spots missing.")

        # check and validate filtered decoded spots
        if self._datastore_state["FilteredSpots"]:
            filtered_path = self._decoded_root_path / Path(
                "all_tiles_filtered_decoded_features.parquet"
            )

            if not (filtered_path.exists()):
                raise Exception("filtered decoded spots missing.")

        if self._datastore_state["RefinedSpots"]:
            baysor_spots_path = (
                self._segmentation_root_path
                / Path("baysor")
                / Path("baysor_filtered_genes.parquet")
            )

            if not (baysor_spots_path.exists()):
                raise Exception("Baysor filtered decoded spots missing.")

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
                raise Exception("mtx output missing.")

        try:
            self._baysor_path = Path(str(self._datastore_state["BaysorPath"]))
            self._baysor_options = Path(str(self._datastore_state["BaysorOptions"]))
            self._julia_threads = int(self._datastore_state["JuliaThreads"])
        except Exception:
            self._baysor_path = r""
            self._baysor_options = r""
            self._julia_threads = 1

    def load_codebook_parsed(
        self,
    ) -> Optional[tuple[Collection[str], ArrayLike]]:
        """Load and split codebook into gene_ids and codebook matrix."""

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
        except Exception:
            print("Error parsing codebook.")
            return None

    def initialize_tile(
        self,
        tile: Union[int, str],
    ):
        """Initialize directory structure for a tile"""

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
            polyDT_tile_path = self._polyDT_root_path / Path(tile_id)
            polyDT_tile_path.mkdir()
            for round_idx, round_id in enumerate(self._round_ids):
                polyDT_round_path = polyDT_tile_path / Path(round_id + ".zarr")
                polyDT_round_path.mkdir()
                polydt_round_attrs_path = polyDT_round_path / Path(".zattrs")
                round_attrs = {
                    "bit_linker": self._experiment_order.to_numpy()[round_idx, 1:]
                    .astype(int)
                    .tolist(),
                }
                self._save_to_json(round_attrs, polydt_round_attrs_path)
        except Exception:
            print("Error creating polyDT tile. Does it exist already?")

        try:
            readout_tile_path = self._readouts_root_path / Path(tile_id)
            readout_tile_path.mkdir()
            for bit_idx, bit_id in enumerate(self._bit_ids):
                readout_bit_path = readout_tile_path / Path(bit_id + ".zarr")
                readout_bit_path.mkdir()
                readout_bit_attrs_path = readout_bit_path / Path(".zattrs")
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
        except Exception as e:
            print(e)
            print("Error creating readout tile. Does it exist already?")

    def load_local_bit_linker(
        self,
        tile: Union[int, str],
        round: Union[int, str],
    ) -> Optional[Sequence[int]]:
        """Load readout bits linked to fidicual round for one tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            return attributes["bits"][1:]
        except Exception:
            print(tile_id, round_id)
            print("Bit linker attribute not found.")
            return None

    def save_local_bit_linker(
        self,
        bit_linker: Sequence[int],
        tile: Union[int, str],
        round: Union[int, str],
    ):
        """Save readout bits linked to fidicual round for one tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            attributes["bits"] = bit_linker
            self._save_to_json(attributes, zattrs_path)
        except Exception:
            print(tile_id, round_id)
            print("Error writing bit linker attribute.")
            return None

    def load_local_round_linker(
        self,
        tile: Union[int, str],
        bit: Union[int, str],
    ) -> Optional[Sequence[int]]:
        """Load fidicual round linked to readout bit for one tile."""

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
            zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(bit_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            return int(attributes["round_linker"])
        except Exception:
            print(tile_id, bit_id)
            print("Round linker attribute not found.")
            return None

    def save_local_round_linker(
        self,
        round_linker: int,
        tile: Union[int, str],
        bit: Union[int, str],
    ):
        """Save fidicual round linker attribute to readout bit for one tile."""

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
            zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(bit_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            attributes["round"] = int(round_linker)
            self._save_to_json(attributes, zattrs_path)
        except Exception:
            print(tile_id, bit_id)
            print("Error writing round linker attribute.")
            return None

    def load_local_stage_position_zyx_um(
        self,
        tile: Union[int, str],
        round: Union[int, str],
    ) -> Optional[ArrayLike]:
        """Load tile stage position for one tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            return np.asarray(attributes["stage_zyx_um"], dtype=np.float32)
        except Exception:
            print(tile_id, round_id)
            print("Stage position attribute not found.")
            return None

    def save_local_stage_position_zyx_um(
        self,
        stage_zyx_um: ArrayLike,
        tile: Union[int, str],
        round: Union[int, str],
    ):
        """Save tile stage position for one tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            attributes["stage_zyx_um"] = stage_zyx_um.tolist()
            self._save_to_json(attributes, zattrs_path)
        except Exception:
            print(tile_id, round_id)
            print("Error writing stage position attribute.")
            return None

    def load_local_wavelengths_um(
        self,
        tile: Union[int, str],
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
    ) -> Optional[tuple[float, float]]:
        """Load wavelengths for fidicual OR readout bit for one tile."""

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
            zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
            )

        try:
            attributes = self._load_from_json(zattrs_path)
            ex_wavelength_um = attributes["excitation_um"]
            em_wavelength_um = attributes["emission_um"]
            return (ex_wavelength_um, em_wavelength_um)
        except Exception:
            print("Wavelength attributes not found.")
            return None

    def save_local_wavelengths_um(
        self,
        wavelengths_um: tuple[float, float],
        tile: Union[int, str],
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
    ) -> Optional[tuple[float, float]]:
        """Save wavelengths for fidicual OR readout bit for one tile."""

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
            zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
            )

        try:
            attributes = self._load_from_json(zattrs_path)
            attributes["excitation_um"] = float(wavelengths_um[0])
            attributes["emission_um"] = float(wavelengths_um[1])
            self._save_to_json(attributes, zattrs_path)
        except Exception:
            print("Error writing wavelength attributes.")
            return None

    def load_local_corrected_image(
        self,
        tile: Union[int, str],
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
        return_future: Optional[bool] = True,
    ) -> Optional[ArrayLike]:
        """Load gain and offset corrected image for fiducial OR readout bit for one tile."""

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
                / Path(local_id + ".zarr")
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
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path("corrected_data")
            )

        if not Path(current_local_zarr_path).exists():
            print("Corrected image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<u2"
            corrected_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            return corrected_image
        except Exception as e:
            print(e)
            print("Error loading corrected image.")
            return None

    def save_local_corrected_image(
        self,
        image: ArrayLike,
        tile: Union[int, str],
        gain_correction: bool = True,
        hotpixel_correction: bool = True,
        shading_correction: bool = False,
        psf_idx: int = 0,
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
        return_future: Optional[bool] = False,
    ):
        """Save gain and offset corrected image."""

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
                / Path(local_id + ".zarr")
                / Path("corrected_data")
            )
            current_local_zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
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
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path("corrected_data")
            )
            current_local_zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
            )

        try:
            self._save_to_zarr_array(
                image,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            attributes = self._load_from_json(current_local_zattrs_path)
            attributes["gain_correction"] = (gain_correction,)
            attributes["hotpixel_correction"] = (hotpixel_correction,)
            attributes["shading_correction"] = (shading_correction,)
            attributes["psf_idx"] = psf_idx
            self._save_to_json(attributes, current_local_zattrs_path)
        except Exception as e:
            print(e)
            print("Error saving corrected image.")
            return None

    def load_local_rigid_xform_xyz_px(
        self,
        tile: Union[int, str],
        round: Union[int, str],
    ) -> Optional[ArrayLike]:
        """Load calculated rigid registration transform for one round and tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            rigid_xform_xyz_px = np.asarray(
                attributes["rigid_xform_xyz_px"], dtype=np.float32
            )
            return rigid_xform_xyz_px
        except Exception:
            print(tile_id, round_id)
            print("Rigid transform mapping back to first round not found.")
            return None

    def save_local_rigid_xform_xyz_px(
        self,
        rigid_xform_xyz_px: ArrayLike,
        tile: Union[int, str],
        round: Union[int, str],
    ) -> Optional[ArrayLike]:
        """Save calculated rigid registration transform for one round and tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(round_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            attributes["rigid_xform_xyz_px"] = rigid_xform_xyz_px.tolist()
            self._save_to_json(attributes, zattrs_path)
        except Exception:
            print("Error writing rigid transform attribute.")
            return None

    def load_coord_of_xform_px(
        self,
        tile: Optional[Union[int, str]],
        round: Optional[Union[int, str]],
        return_future: Optional[bool] = True,
    ) -> Optional[tuple[ArrayLike, ArrayLike]]:
        """Local fidicual optical flow matrix for one round and tile."""

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

        current_local_zarr_path = str(
            self._polyDT_root_path
            / Path(tile_id)
            / Path(round_id + ".zarr")
            / Path("of_xform_px")
        )
        zattrs_path = str(
            self._polyDT_root_path
            / Path(tile_id)
            / Path(round_id + ".zarr")
            / Path(".zattrs")
        )

        if not Path(current_local_zarr_path).exists():
            print("Optical flow transform mapping back to first round not found.")
            return None

        try:
            compressor = {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": 2,
            }
            spec_of = {
                "driver": "zarr",
                "kvstore": None,
                "metadata": {"compressor": compressor},
                "open": True,
                "assume_metadata": False,
                "create": True,
                "delete_existing": False,
            }
            spec_of["metadata"]["dtype"] = "<f4"
            of_xform_px = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                spec_of.copy(),
                return_future,
            )
            attributes = self._load_from_json(zattrs_path)
            downsampling = np.asarray(
                attributes["opticalflow_downsampling"], dtype=np.float32
            )

            return of_xform_px, downsampling
        except Exception as e:
            print(e)
            print("Error loading optical flow transform.")
            return None

    def save_coord_of_xform_px(
        self,
        of_xform_px: ArrayLike,
        tile: Union[int, str],
        downsampling: Sequence[float],
        round: Union[int, str],
        return_future: Optional[bool] = False,
    ):
        """Save fidicual optical flow matrix for one round and tile."""

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
        current_local_zarr_path = str(
            self._polyDT_root_path
            / Path(tile_id)
            / Path(local_id + ".zarr")
            / Path("of_xform_px")
        )
        current_local_zattrs_path = str(
            self._polyDT_root_path
            / Path(tile_id)
            / Path(local_id + ".zarr")
            / Path(".zattrs")
        )

        try:
            compressor = {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": 2,
            }
            spec_of = {
                "driver": "zarr",
                "kvstore": None,
                "metadata": {"compressor": compressor},
                "open": True,
                "assume_metadata": False,
                "create": True,
                "delete_existing": False,
            }
            self._save_to_zarr_array(
                of_xform_px,
                self._get_kvstore_key(current_local_zarr_path),
                spec_of.copy(),
                return_future,
            )
            attributes = self._load_from_json(current_local_zattrs_path)
            attributes["opticalflow_downsampling"] = downsampling
            self._save_to_json(attributes, current_local_zattrs_path)
        except Exception:
            print("Error saving optical flow transform.")
            return None

    def load_local_registered_image(
        self,
        tile: Union[int, str],
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
        return_future: Optional[bool] = True,
    ) -> Optional[ArrayLike]:
        """Local registered, deconvolved image for fidiculial OR readout bit for one tile."""

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
                / Path(local_id + ".zarr")
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
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path("registered_decon_data")
            )

        if not Path(current_local_zarr_path).exists():
            print("Registered deconvolved image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<u2"
            registered_decon_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            return registered_decon_image
        except Exception as e:
            print(e)
            print("Error loading registered deconvolved image.")
            return None

    def save_local_registered_image(
        self,
        registered_image: ArrayLike,
        tile: Union[int, str],
        deconvolution: bool = True,
        round: Optional[Union[int, str]] = None,
        bit: Optional[Union[int, str]] = None,
        return_future: Optional[bool] = False,
    ):
        """Save registered, deconvolved image."""

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
                / Path(local_id + ".zarr")
                / Path("registered_decon_data")
            )
            current_local_zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
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
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path("registered_decon_data")
            )
            current_local_zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
            )

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<u2"
            self._save_to_zarr_array(
                registered_image,
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            attributes = self._load_from_json(current_local_zattrs_path)
            attributes["deconvolution"] = deconvolution
            self._save_to_json(attributes, current_local_zattrs_path)
        except Exception:
            print("Error saving corrected image.")
            return None

    def load_local_ufish_image(
        self,
        tile: Union[int, str],
        bit: Union[int, str],
        return_future: Optional[bool] = True,
    ) -> Optional[ArrayLike]:
        """Load readout bit U-FISH prediction image for one tile."""

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
            / Path(bit_id + ".zarr")
            / Path("registered_ufish_data")
        )

        if not Path(current_local_zarr_path).exists():
            print("U-FISH prediction image not found.")
            return None

        try:
            spec = self._zarrv2_spec.copy()
            spec["metadata"]["dtype"] = "<f4"
            registered_ufish_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                spec,
                return_future,
            )
            return registered_ufish_image
        except Exception as e:
            print(e)
            print("Error loading U-FISH image.")
            return None

    def save_local_ufish_image(
        self,
        ufish_image: ArrayLike,
        tile: Union[int, str],
        bit: Union[int, str],
        return_future: Optional[bool] = False,
    ):
        """Save U-FISH prediction image."""

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
                / Path(local_id + ".zarr")
                / Path("registered_ufish_data")
            )

        try:
            self._save_to_zarr_array(
                ufish_image,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
        except Exception as e:
            print(e)
            print("Error saving U-Fish image.")
            return None

    def load_local_ufish_spots(
        self,
        tile: Union[int, str],
        bit: Union[int, str],
    ) -> Optional[pd.DataFrame]:
        """Load U-FISH spot localizations and features for one tile."""

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

        current_ufish_localizations_path = (
            self._ufish_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )

        if not current_ufish_localizations_path.exists():
            print("U-FISH localizations not found.")
            return None
        else:
            ufish_localizations = self._load_from_parquet(
                current_ufish_localizations_path
            )
            return ufish_localizations

    def save_local_ufish_spots(
        self,
        spot_df: pd.DataFrame,
        tile: Union[int, str],
        bit: Union[int, str],
    ):
        """Save U-FISH localizations and features."""

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

        if not (self._ufish_localizations_root_path / Path(tile_id)).exists():
            (self._ufish_localizations_root_path / Path(tile_id)).mkdir()

        current_ufish_localizations_path = (
            self._ufish_localizations_root_path
            / Path(tile_id)
            / Path(bit_id + ".parquet")
        )

        try:
            self._save_to_parquet(spot_df, current_ufish_localizations_path)
        except Exception as e:
            print(e)
            print("Error saving U-FISH localizations.")
            return None

    def load_global_coord_xforms_um(
        self,
        tile: Union[int, str],
    ) -> Optional[tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """Load global registration transform for one tile."""

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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(self._round_ids[0] + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)
            return (affine_zyx_um, origin_zyx_um, spacing_zyx_um)
        except Exception:
            print(tile_id, self._round_ids[0])
            print("Global coordinate transforms not found")
            return None, None, None

    def save_global_coord_xforms_um(
        self,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        tile: Union[int, str],
    ) -> None:
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
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(self._round_ids[0] + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            attributes["affine_zyx_um"] = affine_zyx_um.tolist()
            attributes["origin_zyx_um"] = origin_zyx_um.tolist()
            attributes["spacing_zyx_um"] = spacing_zyx_um.tolist()
            self._save_to_json(attributes, zattrs_path)
        except Exception as e:
            print(e)

    def load_global_fidicual_image(
        self,
        return_future: Optional[bool] = True,
    ) -> Optional[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]:
        """Load downsampled, fused fidicual image."""

        current_local_zarr_path = str(
            self._fused_root_path / Path("fused.zarr") / Path("fused_polyDT_iso_zyx")
        )

        if not Path(current_local_zarr_path).exists():
            print("Globally registered, fused image not found.")
            return None

        zattrs_path = str(current_local_zarr_path / Path(".zattrs"))

        try:
            fused_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            attributes = self._load_from_json(zattrs_path)
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)
            return fused_image, affine_zyx_um, origin_zyx_um, spacing_zyx_um
        except Exception:
            print("Error loading globally registered, fused image.")
            return None

    def save_global_fidicual_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        fusion_type: str = "polyDT",
        return_future: Optional[bool] = False,
    ):
        """Save downsampled, fused fidicual image."""

        if fusion_type == "polyDT":
            filename = "fused_polyDT_iso_zyx"
        else:
            filename = "fused_all_channels_zyx"
        current_local_zarr_path = str(
            self._fused_root_path / Path("fused.zarr") / Path(filename)
        )
        current_local_zattrs_path = str(
            self._fused_root_path
            / Path("fused.zarr")
            / Path(filename)
            / Path(".zattrs")
        )

        attributes = {
            "affine_zyx_um": affine_zyx_um.tolist(),
            "origin_zyx_um": origin_zyx_um.tolist(),
            "spacing_zyx_um": spacing_zyx_um.tolist(),
        }
        try:
            self._save_to_zarr_array(
                fused_image.astype(np.uint16),
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            self._save_to_json(attributes, current_local_zattrs_path)
        except Exception:
            print("Error saving fused image.")
            return None
        
    def load_local_decoded_spots(
        self,
        tile: Union[int, str],
    ) -> Optional[pd.DataFrame]:
        """Load decoded spots and features for one tile."""

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
        tile: Union[int, str],
    ) -> None:
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
    ) -> Optional[pd.DataFrame]:
        """Load all decoded and filtered spots."""

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
    ):
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
    ) -> Optional[dict]:
        """Load Cellpose max projection cell outlines."""

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
        return_future: Optional[bool] = True,
    ) -> Optional[ArrayLike]:
        """Load Cellpose max projection, downsampled segmentation image."""

        current_local_zarr_path = str(
            self._segmentation_root_path
            / Path("cellpose")
            / Path("cellpose.zarr")
            / Path("masks_polyDT_iso_zyx")
        )

        if not current_local_zarr_path.exists():
            print("Cellpose prediction on global fused image not found.")
            return None

        try:
            fused_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            return fused_image
        except Exception:
            print("Error loading Cellpose image.")
            return None

    def save_global_cellpose_segmentation_image(
        self,
        cellpose_image: ArrayLike,
        downsampling: Sequence[float],
        return_future: Optional[bool] = False,
    ):
        """Save Cellpose max projection, downsampled segmentation image"""

        current_local_zarr_path = str(
            self._segmentation_root_path
            / Path("cellpose")
            / Path("cellpose.zarr")
            / Path("masks_polyDT_iso_zyx")
        )
        current_local_zattrs_path = str(
            self._segmentation_root_path
            / Path("cellpose")
            / Path("cellpose.zarr")
            / Path("masks_polyDT_iso_zyx")
            / Path(".zattrs")
        )

        attributes = {"downsampling": downsampling}

        try:
            self._save_to_zarr_array(
                cellpose_image,
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec.copy(),
                return_future,
            )
            self._save_to_json(attributes, current_local_zattrs_path)
        except Exception:
            print("Error saving Cellpose image.")
            return None

    def save_spots_prepped_for_baysor(self, prepped_for_baysor_df: pd.DataFrame):
        current_global_filtered_decoded_dir_path = self._datastore_path / Path(
            "all_tiles_filtered_decoded_features"
        )

        if not current_global_filtered_decoded_dir_path.exists():
            current_global_filtered_decoded_dir_path.mkdir()

        current_global_filtered_decoded_path = (
            current_global_filtered_decoded_dir_path / Path("transcripts.parquet")
        )

        self._save_to_parquet(prepped_for_baysor_df, current_global_filtered_decoded_path)
        
    def run_baysor(self):
        import subprocess
        
        baysor_input_path = self._datastore_path / Path("all_tiles_filtered_decoded_features") / Path("transcripts.parquet")
        baysor_output_path = self._datastore_path / Path("segmentation")
        
        julia_threading = r"JULIA_NUM_THREADS="+str(self._julia_threads)+ " "
        preview_baysor_options = r"preview -c " +str(self._baysor_options)
        command = julia_threading + str(self._baysor_path) + " " + preview_baysor_options + " " +\
            str(baysor_input_path) + " -o " + str(baysor_output_path)
                    
        try:
            result = subprocess.run(command, shell=True, check=True)
            print("Baysor finished with return code:", result.returncode)
        except subprocess.CalledProcessError as e:
            print("Baysor failed with:", e)
        
        # first try to run Baysor assuming that prior segmentations are present               
        try:
            run_baysor_options = r"run -p -c " +str(self._baysor_options)
            command = julia_threading + str(self._baysor_path) + " " + run_baysor_options + " " +\
                str(baysor_input_path) + " -o " + str(baysor_output_path) + " --count-matrix-format tsv :cell_id"
            result = subprocess.run(command, shell=True, check=True)
            print("Baysor finished with return code:", result.returncode)
        except subprocess.CalledProcessError:
            # then fall back and run without prior segmentations.
            # IMPORTANT: the .toml file has to be defined correctly for this to work!
            try:
                run_baysor_options = r"run -p -c " +str(self._baysor_options)
                command = julia_threading + str(self._baysor_path) + " " + run_baysor_options + " " +\
                    str(baysor_input_path) + " -o " + str(baysor_output_path) + " --count-matrix-format tsv"
                result = subprocess.run(command, shell=True, check=True)
                print("Baysor finished with return code:", result.returncode)
            except subprocess.CalledProcessError as e:
                print("Baysor failed with:", e)
                
    def load_global_baysor_filtered_spots(
        self,
    ) -> Optional[pd.DataFrame]:
        """Load Baysor re-assigned decoded RNA."""

        current_baysor_spots_path = (
            self._segmentation_root_path
            / Path("segmentation.csv")
        )

        if not current_baysor_spots_path.exists():
            print("Baysor filtered genes not found.")
            return None
        else:
            baysor_filtered_genes = self._load_from_csv(current_baysor_spots_path)
            return baysor_filtered_genes

    def load_global_baysor_outlines(
        self,
    ) -> Optional[dict]:
        """Load Baysor cell outlines."""

        current_baysor_outlines_path = (
            self._segmentation_root_path /  Path("segmentation_polygons_3d.json")
        )

        if not current_baysor_outlines_path.exists():
            print("Baysor outlines not found.")
            return None
        else:
            baysor_outlines = self._load_from_microjson(current_baysor_outlines_path)
            return baysor_outlines
        
    def save_mtx(self):
        from merfish3danalysis.utils._dataio import create_mtx

        baysor_output_path = self._datastore_path / Path("segmentation") / Path("segmentation.csv")
        mtx_output_path = self._datastore_path / Path("mtx_output")
        
        create_mtx(
            baysor_output_path=baysor_output_path,
            output_dir_path=mtx_output_path,
        )
