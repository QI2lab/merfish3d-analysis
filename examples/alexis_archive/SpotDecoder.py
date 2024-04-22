"""
SpotDecoder: Decode spots identity from localization data of combinatorial FISH experiments.

2024/01 - Doug Shepherd. Rewrite of API to match qi2lab MERFISH file format v1.0.
          Adapted to use simplified & improved translation/optical flow registration API.
          Rework chunk-wise calculations because widefield data is acquired in tiles.
2023/10 - Alexis Coullomb
"""

from pathlib import Path
import gc
from typing import Union, Optional, Dict
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import dask
from dask.distributed import Client, wait
import multiprocessing
from wf_merfish.postprocess._decode import transform_bit_data, optimize_spots, decode_optimized_chunks
from wf_merfish.postprocess._registration import warp_coordinates
from wf_merfish.postprocess.DataRegistration import DataRegistration
from wf_merfish.postprocess.DataLocalization import DataLocalization
from wf_merfish.utils._dataio import load_localizations
import zarr

class SpotDecoder:
    def __init__(
            self,
            dataset_path: Union[str, Path],
            loading_params: Optional[Dict] = None,
            preprocessing_params: Optional[Dict] = None,
            decoding_params: Optional[Dict] = None,
            barcode_filter_params: Optional[Dict] = None,
            output_path: Optional[Union[str, Path]] = None,
            ):
        """
        Perform decoding for MERFISH localization data.

        Parameters
        ----------
        dataset_Path : Union[str, Path] 
            Path to dataset
        loading_params: Optional[Dict]
            Localization loading parameters.
        preprocessing_params: Optional[Dict]
            Preprocessing parameters.
        decoding_params: Optional[Dict]
            Decoding parameters.
        barcode_filter_params: Optional[Dict]
            Barcode filtering parameters.
        output_path: Optional[Union[str,Path]]
            Output path.
        """

        # TODO: all setters and getters

        self.dataset_path = dataset_path
        if output_path is None:
            self.output_path = dataset_path / Path('decoded')
        else:
            self.output_path = output_path

        self._col_zyx = ['z','y','x']
        self._col_xyz = ['x','y','z']

        self._parse_dataset()
        self.setup_preprocessing_parameters(preprocessing_params)
        self.setup_loading_parameters(loading_params)
        self.setup_decoding_parameters(decoding_params)

    def _parse_dataset(self):
        """
        Parse dataset to discover/load registrations and spot localizations
        """

        calibrations_dir_path = self._dataset_path / Path ("calibrations.zarr")
        polyDT_dir_path = self._dataset_path / Path("polyDT")
        readout_dir_path = self._dataset_path / Path("readouts")
        self._tile_ids = [entry.name for entry in polyDT_dir_path.iterdir() if entry.is_dir()]
        self._num_tiles = len(self._tile_ids)

        self._voxel_spacing_tile = {}
        self._rigid_xforms = {}
        self._of_xforms = {}
        self._localization_paths = {}
        self._bit_round_linker = {}

        # parse calibrations directory for exp_order and codebook
        calibrations_zarr = zarr.open(calibrations_dir_path, mode='r')
        self._codebook = np.asarray(calibrations_zarr['codebook'])

        del calibrations_zarr
        gc.collect()

        # parse polyDT directory for voxel size and registrations
        for tile_idx, tile_id in enumerate(self._tile_ids):
            data_register_factory = DataRegistration(dataset_path=polyDT_dir_path,
                                                        overwrite_registered=False,
                                                        perform_optical_flow=True,
                                                        tile_idx=tile_idx)
            data_register_factory.load_rigid_registrations()
            data_register_factory.load_opticalflow_registrations()
            
            self._voxel_spacing_tile[tile_id] = data_register_factory._voxel_size

            for r_idx, round_id in enumerate (self._round_ids):
                if r_idx > 0:
                    self._rigid_xforms[tile_id][round_id] = data_register_factory.rigid_xform[r_idx]
                    if data_register_factory._has_of_registrations:
                        self._of_xforms[tile_id][round_id] = data_register_factory.of_xform[r_idx]

            del data_register_factory
            gc.collect()

        # parse readouts directory for localization results
        for tile_idx, tile_id in enumerate(self._tile_ids):
            data_localization_factory = DataLocalization(dataset_path=readout_dir_path,
                                                         tile_idx=tile_idx)
            
            self._bit_ids = data_localization_factory._bit_ids

            for bit_idx, bit_id in enumerate (self._bit_ids):
                self._localization_paths[tile_id][bit_id] = data_localization_factory.localization_path[bit_idx]
                self._bit_round_linker[tile_id][bit_id] = data_localization_factory.bit_round_linker[bit_idx]

            del data_localization_factory
            gc.collect()
            
    def setup_loading_parameters(self,loading_params):
        """
        Setup loading parameters.

        Parameters
        ----------
        loading_params: Dict
            Dictionary of parameters for data loading.
        """
        self._load_fitted_var = True
        self._load_candidates = False
        self._load_filter_conditions = False
        self._filter_inplace = True # use filtered localizations only

        # update parameters if provided by user
        if loading_params is not None:
            for key, val in loading_params.items():
                setattr(self, key, val)
    
    def setup_preprocessing_parameters(self,preprocessing_params):
        """
        Setup localization parameters.

        Parameters
        ----------
        preprocessing_params: Dict
            Dictionary of parameters for spot preprocessing.
        """
        # for spots amplitude
        self.threshold_amplitudes = False
        self.amp_thresh = None
        self.scale_amp_per_round = True

        # update parameters if provided by user
        if preprocessing_params is not None:
            for key, val in preprocessing_params.items():
                setattr(self, key, val)

    def setup_decoding_parameters(self, decoding_params):
        """
        Setup decoding algorithms parameters
        TO DO: Rework once new chunking strategy is in place.

        Parameters
        ----------
        preprocessing_params: Dict
            Dictionary of parameters for decoding algorithm.
        """
        # create default parameters
        self.crop_ROI = False
        self.decode_per_chunk = True
        self.use_dask = False
        self.n_cores = None
        self.n_chunks = 10
        self.filter_localizations = False # False if use candidates
        # z/x/y dispersion, mean amplitude, std amplitude, sequence error, selection size
        self.weights = np.array([2, 1, 0, 10, 1])
        self.dist_params = 0.5
        self.bcd_per_spot_params = {'w0': 1, 'b': 16, 'weight': 0.333}
        self.max_positive_bits = 6
        self.propose_method = 'single_step'
        self.initialize = 'maxloss'
        self.trim_network = True
        self.history = False
        self.return_extra = False
        self.return_contribs = False
        self.return_barcodes_loss = False
        self.file_exist = 'skip'
        self.verbose = 2

        # update parameters if provided by user
        if decoding_params is not None:
            for key, val in decoding_params.items():
                setattr(self, key, val)

    def load_localizations(self):
        """
        Load localization results.
        """
        self.localizations = load_localizations(
            self._localization_paths, 
            load_fitted_var=self.load_fitted_var, 
            load_candidates=self.load_candidates, 
            load_filter_conditions=self.load_filter_conditions,
            filter_inplace=self.filter_inplace, 
            )
        self.n_spots = len(self.localizations['detected_coords'])

        # ------ Data preprocessing ------
        
        # Coordinates transformation
        self.localizations['detected_coords'] = warp_coordinates(
            coords=self.localizations['detected_coords'],
            bit_order = self._bit_round_linker,
            tile_ids = self._tile_ids,
            bit_ids = self._bit_ids,
            translation_transform=self._rigid_xforms,
            displacement_field_transform=self._of_xforms,
            )

        # 3D coordinates of localized spots
        self.coords = self.localizations['detected_coords'][self._col_zyx].values
        # Fitted variables during localization, currently the model uses `amplitude` for the optimization step
        self.fitted_vars = self.localizations['all_fit_vars']['amplitude'].values.reshape((-1, 1))
        # Bit indices of each spot
        self.spots_bit_idx = self.locaz['detected_coords']['bit_idx'].values
        # Additional variables used to filter decoded spots
        self.col_spot_stats = ['amplitude']#, 'sigma_xy', 'sigma_z'] #, 'background']
        self.all_fit_vars = self.locaz['all_fit_vars'].loc[:, self.col_spot_stats]

        if self.threshold_amplitudes:
            new_fit_vars = []
            new_coords = []
            new_bit_idxs = []
            new_all_fit_vars = []
            for bit_idx in np.unique(self.spots_bit_idx):
                # first select data from a given bit index
                select = self.spots_bit_idx == bit_idx
                bit_fit_vars = self.fitted_vars[select, 0]
                bit_coords  = self.coords[select]
                bit_spots = self.spots_bit_idx[select]
                bit_all_fit_vars = self.all_fit_vars.loc[select, :]
                # then select this subset data given an amplitude threshold
                select = bit_fit_vars < self.amp_thresh[bit_idx]
                new_fit_vars.append(bit_fit_vars[select])
                new_coords.append(bit_coords[select])
                new_bit_idxs.append(bit_spots[select])  
                new_all_fit_vars.append(bit_all_fit_vars.loc[select, :]) 
            # stack selected data at different bit indices
            coords = np.vstack(new_coords)
            fitted_vars = np.hstack(new_fit_vars).reshape((-1, 1))
            spots_bit_idx = np.hstack(new_bit_idxs)
            all_fit_vars = pd.concat(new_all_fit_vars, axis=0, ignore_index=True)
            fitted_vars = all_fit_vars['amplitude'].values
            del new_coords, new_fit_vars, new_bit_idxs, new_all_fit_vars, bit_fit_vars, bit_coords, bit_spots, bit_all_fit_vars

            
        if self.scale_amp_per_round:
            for bit_idx in range(self.n_bits):
                select = self.spots_bit_idx == bit_idx
                raw_amps = self.fitted_vars[select, :]
                self.fitted_vars[select] = transform_bit_data(raw_amps)

        self.fitted_vars = self.fitted_vars.reshape((-1, 1))
        self.spot_ids = np.arange(len(self.coords))

        if self.decode_per_chunk:
            # create chunk's boundaries  /!\ quick and dirty way
            # should pass the chunking parameters to the function
            self.z_min, self.y_min, self.x_min = self.coords.min(axis=0)
            self.z_max, self.y_max, self.x_max = self.coords.max(axis=0)
            # we chunk in only one direction
            self.chunk_length = (self.y_max - self.y_min) / self.n_chunks
            # overlap = (y_max - y_min) / 1000
            self.overlap = self.dist_params * 2

    # TO DO: Rework this to apply Dask across tiles instead of OPM chunks
 
    def decode(self):
        """
        Decode localizations for all tiles in parallel.
        """

        self.start_time = datetime.now()
        if self.use_dask and self.decode_per_chunk:
            if self.n_cores is None:
                self.n_cores = multiprocessing.cpu_count() # find a wa to 
            client = Client(
                n_workers=self.n_cores, 
                threads_per_worker=1,
                )
            print('Dask client dashbord URL:', client.dashboard_link)


        # rework idea for Dask
        # generate zyx coordinate arrays that match image size
        # generate blocks from this array that are coordinates
        # create a workers with chunks by logically indexing into localizations array
        # have workers return decoded array

        # alternatively, depending on memory needs...
        # estimate max memory needed for each tile
        # pass all spots for one tile to a dask worker. decode tiles in parallel. 

        config_errs = {}
        for y_idx in self.y_idxs:
            config_errs[y_idx] = {}
            select_y = self.locaz['detected_coords']['y_idx'] == y_idx
            for z_idx in self.z_idxs:
                config_errs[y_idx][z_idx] = {self.n_cores}
                select_z = self.locaz['detected_coords']['z_idx'] == z_idx
                select = np.logical_and(select_y, select_z)
                
                if not self.decode_per_chunk:
                                
                    spot_ids = np.arange(select.sum())
                    
                    # TO DO: check if this can use coords and map_overlap like I did in spots3d?
                    self.results =  optimize_spots(
                        coords=self.coords[select, :], 
                        fit_vars=self.fitted_vars[select, :], 
                        spot_ids=spot_ids, 
                        spot_rounds=self.spots_bit_idx[select], 
                        dist_params=self.dist_params, 
                        weights=self.weights,
                        err_corr_dist=1,
                        bcd_per_spot_params=self.bcd_per_spot_params,
                        max_positive_bits=self.max_positive_bits,
                        codebook=self._codebook,
                        propose_method=self.propose_method, 
                        initialize=self.initialize, 
                        trim_network=self.trim_network, 
                        history=self.history,
                        return_extra=self.return_extra,
                        return_contribs=self.return_contribs, 
                        return_barcodes_loss=self.return_barcodes_loss, 
                        verbose=2,
                        )
                    if self.output_path is not None:
                        joblib.dump(self.results, self.output_path)
                else:
                    # analyse per chunk
                    if self.use_dask:
                        parallel_exec = []
                        # make sure we free memory
                        client.restart()
                    for chunk_id in range(self.n_chunks):
                        # /!\ I've mixed up x and y, the code works, but I could correct the convention...
                        # z_lim = z_min + (z_max - z_min) / 1
                        y_lim_min = self.y_min + self.chunk_length * chunk_id
                        y_lim_max = self.y_min + self.chunk_length * (chunk_id + 1) + self.overlap
                        # x_lim = x_min + (x_max - x_min) / 1
                        # conditions on other borders are already fulfilled
                        y_trim_max = self.y_min + self.chunk_length * (chunk_id + 1)

                        coords_lim = {
                            'z_lim_min': None,
                            'z_lim_max': None,
                            'y_lim_min': y_lim_min,
                            'y_lim_max': y_lim_max,
                            'x_lim_min': None,
                            'x_lim_max': None,
                            'y_trim_max': y_trim_max,
                        }
                        extra_str = f'_y_idx-{y_idx}_z_idx-{z_idx}'

                        # actual decoding
                        if self.use_dask:
                            parallel_exec.append(dask.delayed(decode_optimized_chunks)(
                                chunk_id=chunk_id, 
                                dir_save=self.output_path, 
                                coords=self.coords[select, :], 
                                coords_lim=coords_lim,
                                fitted_vars=self.fitted_vars[select, :], 
                                spot_rounds=self.spots_bit_idx[select], 
                                codebook=self.codebook, 
                                dist_params=self.dist_params, 
                                weights=self.weights,
                                err_corr_dist=1,
                                bcd_per_spot_params=self.bcd_per_spot_params,
                                max_positive_bits=self.max_positive_bits,
                                propose_method=self.propose_method, 
                                initialize=self.initialize, 
                                extra_str=extra_str,
                                trim_network=self.trim_network,
                                file_exist=self.file_exist,
                                verbose=0,
                                ))
                        else:
                            decode_optimized_chunks(
                                chunk_id=chunk_id, 
                                dir_save=self.output_path, 
                                coords=self.coords[select, :], 
                                coords_lim=coords_lim,
                                fitted_vars=self.fitted_vars[select, :], 
                                spot_rounds=self.spots_bit_idx[select], 
                                codebook=self.codebook, 
                                dist_params=self.dist_params, 
                                weights=self.weights,
                                err_corr_dist=1,
                                bcd_per_spot_params=self.bcd_per_spot_params,
                                max_positive_bits=self.max_positive_bits,
                                propose_method=self.propose_method, 
                                initialize=self.initialize, 
                                extra_str=extra_str,
                                trim_network=self.trim_network,
                                file_exist=self.file_exist,
                                verbose=1,
                                )

                    if self.use_dask:
                        persisted_parallel_exec = dask.persist(*parallel_exec)
                        for pex in persisted_parallel_exec:
                            try:
                                wait(pex)
                            except Exception:
                                pass

        print("Elapsed time:", datetime.now() - self.start_time)
        print(self.time_stamp(), '\n\nSpot decoding completed\n\n')

