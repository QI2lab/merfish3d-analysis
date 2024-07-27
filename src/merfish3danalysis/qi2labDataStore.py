"""
DataLoader: Load data from qi2lab MERFISH datastore

Shepherd 2024/07 - initial commit
"""

from typing import Union, Optional, Sequence, Tuple, Collection
from pathlib import Path
from numpy.typing import ArrayLike
import tensorstore as ts
import pandas as pd
import numpy as np
import gc
import json
from itertools import product


class qi2labDataStore:
    def __init__(self, datastore_path: Union[str, Path]):
        """API to qi2lab MERFISH store.

        Parameters
        ----------
        datastore_path : Union[str, Path]
            Path to qi2lab MERFISH store
        """

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
        return getattr(self, "_datastore_state", None)

    @datastore_state.setter
    def datastore_state(self, value: dict):
        if not hasattr(self, "_datastore_state") or self._datastore_state is None:
            self._datastore_state = value
        else:
            self._datastore_state.update(value)

    @property
    def microscope_type(self) -> Optional[str]:
        return getattr(self, "_microscope_type", None)

    @microscope_type.setter
    def microscope_type(self, value: str):
        self._microscope_type = value

    @property
    def camera_model(self) -> Optional[str]:
        return getattr(self, "_camera_model", None)

    @camera_model.setter
    def camera_model(self, value: str):
        self._camera_model = value

    @property
    def num_rounds(self) -> Optional[int]:
        return getattr(self, "_num_rounds", None)

    @num_rounds.setter
    def num_rounds(self, value: int):
        self._num_rounds = value

    @property
    def num_tiles(self) -> Optional[int]:
        return getattr(self, "_num_tiles", None)

    @num_tiles.setter
    def num_tiles(self, value: int):
        self._num_tiles = value

    @property
    def channels_in_data(self) -> Optional[Collection[int]]:
        return getattr(self, "_channels_in_data", None)

    @channels_in_data.setter
    def channels_in_data(self, value: Collection[int]):
        self._channels_in_data = value

    @property
    def tile_overlap(self) -> Optional[float]:
        return getattr(self, "_tile_overlap", None)

    @tile_overlap.setter
    def tile_overlap(self, value: float):
        self._tile_overlap = value

    @property
    def binning(self) -> Optional[int]:
        return getattr(self, "_binning", None)

    @binning.setter
    def binning(self, value: int):
        self._binning = value

    @property
    def e_per_ADU(self) -> Optional[float]:
        return getattr(self, "_e_per_ADU", None)

    @e_per_ADU.setter
    def e_per_ADU(self, value: float):
        self._e_per_ADU = value

    @property
    def na(self) -> Optional[float]:
        return getattr(self, "_na", None)

    @na.setter
    def na(self, value: float):
        self._na = value

    @property
    def ri(self) -> Optional[float]:
        return getattr(self, "_ri", None)

    @ri.setter
    def ri(self, value: float):
        self._ri = value

    @property
    def noise_map(self) -> Optional[ArrayLike]:
        return getattr(self, "_noise_map", None)

    @noise_map.setter
    def noise_map(self, value: ArrayLike):
        self._noise_map = value

    @property
    def channel_shading_maps(self) -> Optional[ArrayLike]:
        return getattr(self, "_shading_maps", None)

    @channel_shading_maps.setter
    def channel_shading_map(self, value: ArrayLike):
        self._shading_maps = value

    @property
    def channel_psfs(self) -> Optional[ArrayLike]:
        return getattr(self, "_psfs", None)

    @channel_psfs.setter
    def channel_psf(self, value: ArrayLike):
        self._psfs = value

    @property
    def experiment_order(self) -> Optional[ArrayLike]:
        return getattr(self, "_experiment_order", None)

    @experiment_order.setter
    def experiment_order(self, value: ArrayLike):
        self._experiment_order = value

    @property
    def codebook(self) -> Optional[pd.DataFrame]:
        return getattr(self, "_codebook", None)

    @codebook.setter
    def codebook(self, value: pd.DataFrame):
        self._codebook = value

    @property
    def voxel_size_zyx_um(self) -> Optional[ArrayLike]:
        return getattr(self, "_voxel_size_zyx_um", None)

    @voxel_size_zyx_um.setter
    def voxel_size_zyx_um(self, value: ArrayLike):
        self._voxel_size_zyx_um = value

    @property
    def tile_ids(self) -> Optional[Collection[str]]:
        return getattr(self, "_tile_ids", None)

    @property
    def rounds_ids(self) -> Optional[Collection[str]]:
        return getattr(self, "_round_ids", None)

    @property
    def bit_ids(self) -> Optional[Collection[str]]:
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
                with open(zattrs_path, "r") as f:
                    attributes = json.load(f)
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
                current_local = ts.open(
                    {
                        **self._zarrv2_spec,
                        "kvstore": self._get_kvstore_key(current_local_zarr_path),
                    }
                ).result()
                self._psfs = current_local.read().result()
            except Exception:
                print("Calibration psfs missing.")

            del current_local_zarr_path, current_local

            current_local_zarr_path = str(
                self._calibrations_zarr_path / Path("noise_map")
            )

            try:
                current_local = ts.open(
                    {
                        **self._zarrv2_spec,
                        "kvstore": self._get_kvstore_key(current_local_zarr_path),
                    }
                ).result()
                self._noise_map = current_local.read().result()
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
                    with open(zattrs_path, "r") as f:
                        attributes = json.load(f)
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
                    current_local = ts.open(
                        {
                            **self._zarrv2_spec,
                            "kvstore": self._get_kvstore_key(current_local_zarr_path),
                        }
                    ).result()
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
                    with open(zattrs_path, "r") as f:
                        attributes = json.load(f)
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
                    current_local = ts.open(
                        {
                            **self._zarrv2_spec,
                            "kvstore": self._get_kvstore_key(current_local_zarr_path),
                        }
                    ).result()
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
                        current_local = ts.open(
                            {
                                **self._zarrv2_spec,
                                "kvstore": self._get_kvstore_key(
                                    current_local_zarr_path
                                ),
                            }
                        ).result()
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
                    current_local = ts.open(
                        {
                            **self._zarrv2_spec,
                            "kvstore": self._get_kvstore_key(current_local_zarr_path),
                        }
                    ).result()
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
                    current_local = ts.open(
                        {
                            **self._zarrv2_spec,
                            "kvstore": self._get_kvstore_key(current_local_zarr_path),
                        }
                    ).result()
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
                    current_local = ts.open(
                        {
                            **self._zarrv2_spec,
                            "kvstore": self._get_kvstore_key(current_local_zarr_path),
                        }
                    ).result()
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
                current_local = ts.open(
                    {
                        **self._zarrv2_spec,
                        "kvstore": self._get_kvstore_key(current_local_zarr_path),
                    }
                ).result()
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
                current_local = ts.open(
                    {
                        **self._zarrv2_spec,
                        "kvstore": self._get_kvstore_key(current_local_zarr_path),
                    }
                ).result()
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

    def load_codebook_parsed(
        self,
    ) -> Optional[Tuple[Collection[str], ArrayLike]]:
        pass

    def load_local_bit_linker(
        self,
        tile_idx: int = 0,
        round_idx: int = 0,
    ) -> Optional[Sequence[int]]:
        pass

    def save_local_bit_linker(
        self,
        bit_linker: Sequence[int],
        tile_idx: int = 0,
        round_idx: int = -1,
    ):
        pass

    def load_local_round_linker(
        self,
        tile_idx: int = 0,
        bit_idx: int = -1,
    ) -> Optional[Sequence[int]]:
        pass

    def save_local_round_linker(
        self,
        round_linker: Sequence[int],
        tile_idx: int = 0,
        bit_idx: int = -1,
    ):
        pass

    def load_local_stage_position_zyx_um(
        self,
        tile_idx: int = 0,
        round_idx: int = -1,
    ) -> Optional[ArrayLike]:
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        else:
            return None

        if not (current_local_zarr_path.exists()):
            return None

        current_local = ts.open(
            {
                **self._zarrv2_spec,
                "kvstore": self._get_kvstore_key(current_local_zarr_path),
            }
        ).result()

        try:
            stage_position_zyx_um = np.asarray(
                current_local.attrs["stage_zyx_um"], dtype=np.float32
            )
        except Exception:
            return None

        del current_local_zarr_path, current_local
        gc.collect()

        return stage_position_zyx_um

    def save_local_stage_position_zyx_um(
        self,
        stage_position_zyx_um: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ):
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        else:
            return

        spec = self._zarrv2_spec.copy()
        if not current_local_zarr_path.exists():
            spec["create"] = True
            spec["delete_existing"] = False

        spec["kvstore"] = self._get_kvstore_key(current_local_zarr_path)
        spec["metadata"]["dtype"] = "<f4"
        spec["metadata"]["shape"] = (1,)
        spec["metadata"]["chunks"] = (1,)

        current_local = ts.open(spec).result()

        current_local.attrs["stage_zyx_um"] = np.array(stage_position_zyx_um).tolist()

        del current_local_zarr_path, current_local, spec
        gc.collect()

    def load_local_wavelengths_um(
        self,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ) -> Optional[Tuple[float, float]]:
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return None

        if not current_local_zarr_path.exists():
            return None

        current_local = ts.open(
            {
                **self._zarrv2_spec,
                "kvstore": self._get_kvstore_key(current_local_zarr_path),
            }
        ).result()

        try:
            excitation_um = float(current_local.attrs["excitation_um"])
            emission_um = float(current_local.attrs["emission_um"])
        except Exception:
            return None

        del current_local_zarr_path, current_local
        gc.collect()

        return (excitation_um, emission_um)

    def save_local_wavelengths_um(
        self,
        wavelengths_um: Tuple[float, float],
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ):
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return

        spec = self._zarrv2_spec.copy()
        if not current_local_zarr_path.exists():
            spec["create"] = True
            spec["delete_existing"] = False

        spec["kvstore"] = self._get_kvstore_key(current_local_zarr_path)
        spec["metadata"]["dtype"] = "<f4"
        spec["metadata"]["shape"] = (1,)
        spec["metadata"]["chunks"] = (1,)

        current_local = ts.open(spec).result()

        current_local.attrs["excitation_um"] = wavelengths_um[0]
        current_local.attrs["emission_um"] = wavelengths_um[1]

        del current_local_zarr_path, current_local, spec
        gc.collect()

    def load_local_corrected_image(
        self,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ) -> Optional[ArrayLike]:
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return None

        if not current_local_zarr_path.exists():
            return None

        current_local = ts.open(
            {
                **self._zarrv2_spec,
                "kvstore": self._get_kvstore_key(current_local_zarr_path),
            }
        ).result()

        try:
            corrected_image = current_local["corrected_data"]
        except Exception:
            return None

        del current_local_zarr_path, current_local
        gc.collect()

        return corrected_image

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
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return

        spec = self._zarrv2_spec.copy()
        if not current_local_zarr_path.exists():
            spec["create"] = True
            spec["delete_existing"] = False

        spec["kvstore"] = self._get_kvstore_key(current_local_zarr_path)
        spec["metadata"]["dtype"] = str(image.dtype)
        spec["metadata"]["shape"] = image.shape
        spec["metadata"]["chunks"] = [1, image.shape[1], image.shape[2]]

        current_local = ts.open(spec).result()

        current_local["corrected_data"][...] = image
        current_local.attrs["gain"] = gain_correction
        current_local.attrs["hotpixel"] = hotpixel_correction
        current_local.attrs["shading"] = shading_correction

        del current_local_zarr_path, current_local, spec
        gc.collect()

        return

    def load_local_coord_rigid_xform_xyz_px(
        self,
        tile_idx: int = 0,
        round_idx: int = 0,
    ) -> ArrayLike:
        pass

    def save_local_coord_rigid_xform_xyz_px(
        self,
        save_rigid_xform_xyz_px: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ) -> None:
        pass

    def load_local_coord_of_xform_3x_px(
        self,
        tile_idx: int = 0,
        round_idx: int = 0,
    ) -> ArrayLike:
        pass

    def save_local_coord_of_xform_3x_px(
        self,
        save_of_xform_3x_px: ArrayLike,
        tile_idx: int = 0,
        round_idx: int = 0,
    ):
        pass

    def load_local_registered_image(
        self,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ) -> Optional[ArrayLike]:
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return None

        if not current_local_zarr_path.exists():
            return None

        current_local = ts.open(
            {
                **self._zarrv2_spec,
                "kvstore": self._get_kvstore_key(current_local_zarr_path),
            }
        ).result()

        try:
            registered_image = current_local["registered_decon_data"]
        except Exception:
            return None

        del current_local_zarr_path, current_local
        gc.collect()

        return registered_image

    def save_local_registered_image(
        self,
        registered_image: ArrayLike,
        deconvolution_run: bool = True,
        tile_idx: int = 0,
        round_idx: Optional[int] = -1,
        bit_idx: Optional[int] = -1,
    ):
        if round_idx > -1:
            current_local_zarr_path = str(
                self._polyDT_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._round_ids[round_idx] + ".zarr")
            )
        elif bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return

        if not current_local_zarr_path.exists():
            return

        spec = self._zarrv2_spec.copy()
        spec["kvstore"] = self._get_kvstore_key(current_local_zarr_path)
        spec["metadata"]["dtype"] = str(registered_image.dtype)
        spec["metadata"]["shape"] = registered_image.shape
        spec["metadata"]["chunks"] = [
            1,
            registered_image.shape[1],
            registered_image.shape[2],
        ]

        current_local = ts.open(spec).result()

        current_local["registered_decon_data"][...] = registered_image

        del current_local_zarr_path, current_local, spec
        gc.collect()

        return

    def load_local_ufish_image(
        self,
        tile_idx: int = 0,
        bit_idx: int = -1,
    ) -> Optional[ArrayLike]:
        if bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return None

        if not current_local_zarr_path.exists():
            return None

        current_local = ts.open(
            {
                **self._zarrv2_spec,
                "kvstore": self._get_kvstore_key(current_local_zarr_path),
            }
        ).result()

        try:
            ufish_image = current_local["registered_ufish_data"]
        except Exception:
            return None

        del current_local_zarr_path, current_local
        gc.collect()

        return ufish_image

    def save_local_ufish_image(
        self,
        ufish_image: ArrayLike,
        tile_idx: int = 0,
        bit_idx: int = -1,
    ):
        if bit_idx > -1:
            current_local_zarr_path = str(
                self._readouts_root_path
                / Path(self._tile_ids[tile_idx])
                / Path(self._bit_ids[bit_idx] + ".zarr")
            )
        else:
            return

        if not current_local_zarr_path.exists():
            return

        spec = self._zarrv2_spec.copy()
        spec["kvstore"] = self._get_kvstore_key(current_local_zarr_path)
        spec["metadata"]["dtype"] = str(ufish_image.dtype)
        spec["metadata"]["shape"] = ufish_image.shape
        spec["metadata"]["chunks"] = [1, ufish_image.shape[1], ufish_image.shape[2]]

        current_local = ts.open(spec).result()

        current_local["registered_ufish_data"][...] = ufish_image

        del current_local_zarr_path, current_local, spec
        gc.collect()

        return

    def load_local_ufish_spots(
        self,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> pd.DataFrame:
        pass

    def save_local_ufish_spots(
        self,
        spot_df: pd.DataFrame,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> None:
        pass

    def load_global_coord_xforms_um(
        self,
        tile_idx: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        pass

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
    ) -> ArrayLike:
        pass

    def save_global_fidicual_image(
        self,
        fused_image: ArrayLike,
        affine_zyx_um: ArrayLike,
        origin_zyx_um: ArrayLike,
        spacing_zyx_um: ArrayLike,
    ) -> None:
        pass

    def load_local_features(
        self,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> pd.DataFrame:
        pass

    def save_tile_features(
        self,
        features_df: pd.DataFrame,
        tile_idx: int = 0,
        bit_idx: int = 0,
    ) -> None:
        pass

    def load_tile_decoded_spots(
        self,
        tile_idx: int = 0,
    ) -> pd.DataFrame:
        pass

    def save_tile_decoded_spots(
        self,
        decoded_df: pd.DataFrame,
        tile_idx: int = 0,
    ) -> None:
        pass

    def load_global_filtered_decoded_spots(
        self,
    ) -> pd.DataFrame:
        pass

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
