"""
DataLoader: Load data from qi2lab MERFISH datastore

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
        self._blosc_compressor = {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 2,
        }
        self._zarrv2_spec = {
            "driver": "zarr",
            "kvstore": None,
            "metadata": {},
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
            current_local = ts.open(
                {
                    **self._zarrv2_spec,
                    "kvstore": self._get_kvstore_key(current_local_zarr_path),
                }
            ).result()
            current_local[...] = value
        except Exception:
            print(r"Could not access calibrations.zarr/noise_map")

    @property
    def channel_shading_maps(self) -> Optional[ArrayLike]:
        """Channel shaiding images."""

        return getattr(self, "_shading_maps", None)

    @channel_shading_maps.setter
    def channel_shading_map(self, value: ArrayLike):
        self._shading_maps = value
        current_local_zarr_path = str(
            self._calibrations_zarr_path / Path("shading_maps")
        )

        try:
            current_local = ts.open(
                {
                    **self._zarrv2_spec,
                    "kvstore": self._get_kvstore_key(current_local_zarr_path),
                }
            ).result()
            current_local[...] = value
        except Exception:
            print(r"Could not access calibrations.zarr/shading_maps")

    @property
    def channel_psfs(self) -> Optional[ArrayLike]:
        """Channel point spread functions (PSF)."""
        return getattr(self, "_psfs", None)

    @channel_psfs.setter
    def channel_psf(self, value: ArrayLike):
        self._psfs = value
        current_local_zarr_path = str(self._calibrations_zarr_path / Path("psf_data"))

        try:
            current_local = ts.open(
                {
                    **self._zarrv2_spec,
                    "kvstore": self._get_kvstore_key(current_local_zarr_path),
                }
            ).result()
            current_local[...] = value
        except Exception:
            print(r"Could not access calibrations.zarr/psf_data")

    @property
    def experiment_order(self) -> Optional[pd.DataFrame]:
        """Round and bit order."""

        return pd.DataFrame(
            getattr(self, "_experiment_order", None), columns=["round", "yellow", "red"]
        )

    @experiment_order.setter
    def experiment_order(self, value: pd.DataFrame):
        self._experiment_order = value
        zattrs_path = self._calibrations_zarr_path / Path(".zattrs")
        calib_zattrs = self._load_from_json(zattrs_path)
        calib_zattrs["exp_order"] = self._experimental_order.values.tolist()
        self._save_to_json(calib_zattrs, zattrs_path)

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
        calib_zattrs["exp_order"] = self._codebook.values.tolist()
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

        self._datastore_path.mkdir()
        self._calibrations_zarr_path = self._datastore_path / Path(r"calibrations.zarr")
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

        # initialize datastore state
        self._datastore_state_json_path = self._datastore_path / Path(
            r"datastore_state.json"
        )
        self._datastore_state = {
            "Version": 0.3,
            "Initialized": True,
            "Calibrations": False,
            "Corrected": False,
            "LocalRegister": False,
            "GlobalRegister": False,
            "Fused": False,
            "SegmentedCells": False,
            "DecodedSpots": False,
            "FilteredSpots": False,
            "RefinedSpots": False,
            "mtxOutput": False,
        }

        with open(self._datastore_path / Path("progress.json"), "w") as json_file:
            json.dump(self._datastore_state, json_file, indent=4)

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

        array_data = current_zarr

        if return_future:
            return array_data.read()
        else:
            return array_data.read().result()

    @staticmethod
    def _save_to_zarr_array(
        array: ArrayLike,
        kvstore: dict,
        spec: dict,
        return_future: bool = True,
    ) -> Optional[ArrayLike]:
        """Save array to zarr using tensorstore.

        Defaults to returning future result.
        """

        spec["Metadata"]["shape"] = array.shape
        spec["Metadata"]["chunks"] = [1, array.shape[1], array.shape[2]]

        current_zarr = ts.open(
            {
                **spec,
                "kvstore": kvstore,
            }
        ).result()

        current_zarr = array

        if return_future:
            return current_zarr
        else:
            current_zarr.result()
            return None

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
        # with open("progress.json", "r") as json_file:
        #     self._datastore_state = json.load(json_file)

        self._datastore_state = {
            "Version": 0.2,
            "Initialized": True,
            "Calibrations": True,
            "Corrected": True,
            "LocalRegistered": True,
            "GlobalRegistered": True,
            "Fused": True,
            "SegmentedCells": True,
            "DecodedSpots": True,
            "FilteredSpots": True,
            "RefinedSpots": True,
            "mtxOutput": False,
        }

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
                "experiment_order",
                "codebook",
            ]
            if self._datastore_state["Version"] == 0.3:
                keys_to_check.append(
                    ["microscope_type", "camera_model", "voxel_size_zyx_um"]
                )
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
                        spec=self._zarrv2_spec,
                    )
                ).result()
            except Exception:
                print("Calibration psfs missing.")

            del current_local_zarr_path

            current_local_zarr_path = str(
                self._calibrations_zarr_path / Path("noise_map")
            )

            try:
                self._noise_map = (
                    self._load_from_zarr_array(
                        kvstore=self._get_kvstore_key(current_local_zarr_path),
                        spec=self._zarrv2_spec,
                    )
                ).result()
            except Exception:
                print("Calibration noise map missing.")

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
                    "voxel_zyx_um",
                    "bits",
                    "exposure_ms",
                    "psf_idx",
                ]

                for key in keys_to_check:
                    if key not in attributes.keys():
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
                        self._zarrv2_spec,
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
                    "stage_zyx_um",
                    "excitation_um",
                    "emission_um",
                    "voxel_zyx_um",
                    "round",
                    "exposure_ms",
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
                        self._zarrv2_spec,
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
                        / Path("of_xform_3x_px")
                    )

                    try:
                        self._check_for_zarr_array(
                            self._get_kvstore_key(current_local_zarr_path),
                            self._zarrv2_spec,
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
                        self._zarrv2_spec,
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
                        self._zarrv2_spec,
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
                        self._zarrv2_spec,
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
                raise Exception(tile_id + " " + bit_id + " ufish localization missing")

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
                    self._get_kvstore_key(current_local_zarr_path), self._zarrv2_spec
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
                    self._get_kvstore_key(current_local_zarr_path), self._zarrv2_spec
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

    # Loading and saving functions

    def load_codebook_parsed(
        self,
    ) -> Optional[tuple[Collection[str], ArrayLike]]:
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
            return attributes["bits"][1:]
        except Exception:
            print(tile_id, round_id)
            print("Local attributes not found")
            return None

    def save_local_bit_linker(
        self,
        bit_linker: Sequence[int],
        tile_idx: int = 0,
        round_idx: int = -1,
    ):
        pass

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

        try:
            zattrs_path = str(
                self._readouts_root_path
                / Path(tile_id)
                / Path(bit_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            return attributes["round"]
        except Exception:
            print(tile_id, bit_id)
            print("Local attributes not found")
            return None

    def save_local_round_linker(
        self,
        round_linker: Sequence[int],
        tile_idx: int = 0,
        bit_idx: int = -1,
    ):
        pass

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
            return np.asarray(attributes["stage_zyx_um"], dtype=np.float32)
        except Exception:
            print(tile_id, round_id)
            print("Local attributes not found")
            return None

    def save_local_stage_position_zyx_um(
        self,
        stage_position_zyx_um: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ):
        pass

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

        try:
            zattrs_path = str(
                self._polyDT_root_path
                / Path(tile_id)
                / Path(local_id + ".zarr")
                / Path(".zattrs")
            )
            attributes = self._load_from_json(zattrs_path)
            ex_wavelength_um = attributes["excitation_um"]
            em_wavelength_um = attributes["emission_um"]
            return (ex_wavelength_um, em_wavelength_um)
        except Exception:
            print("Local attributes not found")
            return None

    def save_local_wavelengths_um(
        self,
        wavelengths_um: tuple[float, float],
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ):
        pass

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
            print("Array does not exist.")
            return None

        try:
            corrected_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            return corrected_image
        except Exception:
            print("Error loading corrected image.")
            return None

    def save_local_corrected_image(
        self,
        image: ArrayLike,
        gain_correction: bool = True,
        hotpixel_correction: bool = True,
        shading_correction: bool = False,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ):
        pass

    def load_local_rigid_xform_xyz_px(
        self,
        tile: Union[int, str],
        round: Union[int, str],
    ) -> ArrayLike:
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
            print("Local attributes not found")
            return None

    def save_local_rigid_xform_xyz_px(
        self,
        save_rigid_xform_xyz_px: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ) -> None:
        pass

    def load_coord_of_xform_px(
        self,
        tile: Optional[Union[int, str]],
        round: Optional[Union[int, str]],
        return_future: Optional[bool] = True,
    ) -> ArrayLike:
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
            / Path("of_xform_3_px")
        )

        if not current_local_zarr_path.exists():
            print("Array does not exist.")
            return None

        try:
            of_xform_px = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            return of_xform_px
        except Exception:
            print("Error loading optical flow transform.")
            return None

    def save_coord_of_xform_px(
        self,
        save_of_xform_px: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ):
        pass

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

        if not current_local_zarr_path.exists():
            print("Array does not exist.")
            return None

        try:
            registered_decon_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            return registered_decon_image
        except Exception:
            print("Error loading registered image.")
            return None

    def save_local_registered_image(
        self,
        registered_image: ArrayLike,
        deconvolution_run: bool = True,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ):
        pass

    def load_local_ufish_image(
        self,
        tile: Union[int, str],
        bit: Union[int, str],
        return_future: Optional[bool] = True,
    ) -> Optional[ArrayLike]:
        """Load readout bit U-Fish prediction image for one tile."""

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

        if not current_local_zarr_path.exists():
            print("Array does not exist.")
            return None

        try:
            registered_ufish_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            return registered_ufish_image
        except Exception:
            print("Error loading ufish image.")
            return None

    def save_local_ufish_image(
        self,
        ufish_image: ArrayLike,
        tile_idx: int = 0,
        bit_idx: int = -1,
    ):
        pass

    def load_local_ufish_spots(
        self,
        tile: Union[int, str],
        bit: Union[int, str],
    ) -> pd.DataFrame:
        """Load U-Fish spot localizations and features for one tile."""

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
            print("Array does not exist.")
            return None
        else:
            ufish_localizations = self._load_from_parquet(
                current_ufish_localizations_path
            )
            return ufish_localizations

    def save_local_ufish_spots(
        self,
        spot_df: pd.DataFrame,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> None:
        pass

    def load_global_coord_xforms_um(
        self,
        tile: Union[int, str],
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Load global registration transform for one tile."""

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
            affine_zyx_um = np.asarray(attributes["affine_zyx_um"], dtype=np.float32)
            origin_zyx_um = np.asarray(attributes["origin_zyx_um"], dtype=np.float32)
            spacing_zyx_um = np.asarray(attributes["spacing_zyx_um"], dtype=np.float32)
            return (affine_zyx_um, origin_zyx_um, spacing_zyx_um)
        except Exception:
            print(tile_id, self._round_ids[0])
            print("Local attributes not found")
            return None

    def save_global_coord_xforms_um(
        self,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
        tile_idx: int = 0,
    ) -> None:
        pass

    def load_global_fidicual_image(
        self,
        return_future: Optional[bool] = True,
    ) -> ArrayLike:
        """Load downsampled, fused fidicual image."""

        current_local_zarr_path = str(
            self._fused_root_path / Path("fused.zarr") / Path("fused_polyDT_iso_zyx")
        )

        if not current_local_zarr_path.exists():
            print("Array does not exist.")
            return None

        try:
            fused_image = self._load_from_zarr_array(
                self._get_kvstore_key(current_local_zarr_path),
                self._zarrv2_spec,
                return_future,
            )
            return fused_image
        except Exception:
            print("Error loading fused image.")
            return None

    def save_global_fidicual_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
    ) -> None:
        pass

    def load_local_decoded_spots(
        self,
        tile: Union[int, str],
    ) -> pd.DataFrame:
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
            print("Array does not exist.")
            return None
        else:
            tile_features = self._load_from_parquet(current_tile_features_path)
            return tile_features

    def save_local_decoded_spots(
        self,
        features_df: pd.DataFrame,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> None:
        pass

    def load_global_filtered_decoded_spots(
        self,
    ) -> pd.DataFrame:
        """Load all decoded and filtered spots."""

        current_global_filtered_decoded_path = self._decoded_root_path / Path(
            "all_tiles_filtered_decoded_features.parquet"
        )

        if not current_global_filtered_decoded_path.exists():
            print("Array does not exist.")
            return None
        else:
            all_tiles_filtered = self._load_from_parquet(
                current_global_filtered_decoded_path
            )
            return all_tiles_filtered

    def save_global_filtered_decoded_spots(
        self,
        filtered_decoded_df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    def load_global_cellpose_centroids(
        self,
    ) -> pd.DataFrame:
        pass

    def save_global_cellpose_centroids(
        self,
        centroids: pd.DataFrame,
    ) -> None:
        pass

    def load_global_cellpose_outlines(
        self,
    ) -> dict:
        pass

    def save_global_cellpose_outlines(
        self,
        outlines: dict,
    ) -> None:
        pass

    def load_global_cellpose_segmentation_image(
        self,
    ) -> ArrayLike:
        pass

    def save_global_cellpose_segmentation_image(
        self,
        cellpose_image: ArrayLike,
    ) -> None:
        pass

    def load_global_baysor_assigned_spots(
        self,
    ) -> pd.DataFrame:
        pass

    def save_global_baysor_assigned_spots(
        self,
        baysor_spots_df: pd.DataFrame,
    ) -> None:
        pass

    def load_global_baysor_outlines(
        self,
    ) -> dict:
        pass

    def save_global_baysor_outlines(self, outlines: dict) -> None:
        pass

    def save_mtx(self, called_spots_df: pd.DataFrame) -> None:
        pass
