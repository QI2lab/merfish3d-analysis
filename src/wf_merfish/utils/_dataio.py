#!/usr/bin/env python

'''
QI2lab OPM suite
Reconstruction tools

Read and write metadata, read raw data, Zarr creation, Zarr conversion
'''

import re
from typing import Dict, Union, List
from numpy.typing import NDArray
import pandas as pd
import numpy as np
from pathlib import Path
from pycromanager import Dataset
from datetime import datetime

def read_metadatafile(fname: Union[str,Path]) -> Dict:
    """
    Read data from csv file consisting of one line giving titles, and the other giving values.
    Return as dictionary

    Parameters
    ----------
    fname: Union[str,Path]
        filename

    Returns
    -------
    metadata: Dict
        metadata dictionary
    """

    scan_data_raw_lines = []

    with open(fname, "r") as f:
        for line in f:
            scan_data_raw_lines.append(line.replace("\n", ""))

    titles = scan_data_raw_lines[0].split(",")

    # convert values to appropriate datatypes
    vals = scan_data_raw_lines[1].split(",")
    for ii in range(len(vals)):
        if re.fullmatch("\d+", vals[ii]):
            vals[ii] = int(vals[ii])
        elif re.fullmatch("\d*.\d+", vals[ii]):
            vals[ii] = float(vals[ii])
        elif vals[ii].lower() == "False".lower():
            vals[ii] = False
        elif vals[ii].lower() == "True".lower():
            vals[ii] = True
        else:
            # otherwise, leave as string
            pass

    # convert to dictionary
    metadata = {}
    for t, v in zip(titles, vals):
        metadata[t] = v

    return metadata

def read_config_file(config_path):
    """
    Read data from csv file consisting of one line giving titles, and the other giving values. Return as dictionary

    :param config_path: Path
        Location of configuration file
    :return dict_from_csv: dict
        instrument configuration metadata
    """

    dict_from_csv = pd.read_csv(config_path, header=None, index_col=0).squeeze("columns").to_dict()

    return dict_from_csv

def read_fluidics_program(program_path):
    """
    Read fluidics program from CSV file as pandas dataframe

    :param program_path: Path
        location of fluidics program

    :return df_fluidics: Dataframe
        dataframe containing fluidics program 
    """

    try:                
        df_fluidics = pd.read_csv(program_path)            
        df_fluidics = df_fluidics[["round", "source", "time", "pump"]]
        df_fluidics.dropna(axis=0, how='any', inplace=True)
        df_fluidics["round"] = df_fluidics["round"].astype(int)
        df_fluidics["pump"] = df_fluidics["pump"].astype(int)

        print("Fluidics program loaded")
    except Exception as e:
        raise Exception("Error in loading fluidics file:\n", e)

    return df_fluidics

def write_metadata(data_dict: Dict, 
                   save_path: Union[str,Path]):
    """
    Write dictionary as CSV file

    Parameters
    ----------
    data_dict: dict
        metadata dictionary
    save_path: Union[str,Path]
        path for file
    
    Returns
    -------
    None

    """
    
    pd.DataFrame([data_dict]).to_csv(save_path)

def return_data_dask(dataset: Dataset,
                     channel_id: str) -> NDArray:
    """
    Return NDTIFF data as a numpy array via dask

    Parameters
    ----------
    dataset: Dataset
        pycromanager dataset object
    channel_axis: str
        channel axis name. One of 'Blue', 'Yellow', 'Red'.

    Returns
    -------
    data: NDArray
        data stack
    """

    data = dataset.as_array(channel=channel_id)
    data = data.compute(scheduler="single-threaded")

    return np.squeeze(data)

def load_localizations(localization_paths: Dict[Union[Path,str]],
                       tile_ids: List[Union[Path,str]],
                       bit_ids: List[Union[Path,str]],
                       load_candidates: bool,
                       load_fitted_var: bool,
                       load_filter_conditions: bool,
                       filter_inplace: bool=False) -> pd.DataFrame:
    """
    Load all localizations from a qi2lab MERFISH dataset.

    Parameters
    ----------
    localization_paths: Dict[Union[Path,str]]
        nested dictionary of path to directory containing localization results
        for [tile][round]     
    tile_ids: List[Union[Path,str]]
        list of tile_ids
    bit_ids: List[Union[Path,str]])
        list of bit ids
    load_candidates: bool 
        flag to load spot candidates
    load_fitted_var: bool
        flag to load fitted variables
    load_filter_conditions: bool
        flag to load filter conditions
    filter_inplace: bool
        flag to filter spots

    Returns
    -------
    all_data: pd.DataFrame
        all localization results 
    """
    all_spots = []
    all_tiles = []
    all_bits = []
    all_fit_vars = []
    all_candidates = []
    all_conditions = []

    for tile_idx, tile_id in enumerate(tile_ids):
        for bit_idx, bit_id in enumerate(bit_ids):
            localization_path = localization_paths[tile_id][bit_id]

            # load coords z / y / x and boolean selector
            spots = pd.read_parquet(localization_path / Path('localized_spots.parquet'))
            # use only selected spots after filtering
            if filter_inplace:
                select = spots['select'].values
                spots = spots.loc[select, ['z', 'y', 'x']].values
            all_spots.append(spots)

            # keep track of bit id across rounds and channels
            all_tiles.extend([tile_idx] * len(spots))
            all_bits.extend([bit_idx] * len(spots))
            
            # load fitted variables
            if load_fitted_var:                    
                fit_vars = pd.read_parquet(localization_path / Path('fitted_variables.parquet'))
                if filter_inplace:
                    fit_vars = fit_vars.loc[select, :]
                all_fit_vars.append(fit_vars)

            # load candidates
            if load_candidates:                    
                candidates = pd.read_parquet(localization_path / Path('localization_candidates.parquet'))
                if filter_inplace:
                    candidates = candidates.loc[select, :]
                all_candidates.append(candidates)

            # load filter conditions
            if load_filter_conditions:                    
                conditions = pd.read_parquet(localization_path / Path('filter_conditions.parquet'))
                if filter_inplace:
                    conditions = conditions.loc[select, :]
                all_conditions.append(conditions)
    
    all_spots = np.vstack(all_spots)
    col_names = ['z', 'y', 'x']
    if not filter_inplace:
        # boolean selector was not discarded
        # /!\ this assumes `select` is the last column /!\
        col_names = col_names + ['select']
    detected_coords = pd.DataFrame(data=all_spots, columns=col_names)
    detected_coords['tile_idx'] = all_tiles
    detected_coords['bit_idx'] = all_bits
    
    all_data = {}
    all_data['detected_coords'] = detected_coords

    if load_fitted_var:
        all_fit_vars = np.vstack(all_fit_vars)
        col_names = ['amplitude', 'x', 'y', 'z', 'sigma_xy', 'sigma_z', 'offset']
        all_fit_vars = pd.DataFrame(data=all_fit_vars, columns=col_names)
        all_data['all_fit_vars'] = all_fit_vars

    if load_candidates:
        all_candidates = np.vstack(all_candidates)
        col_names = ['z', 'y', 'x', 'amplitude']
        all_candidates = pd.DataFrame(data=all_candidates, columns=col_names)
        all_data['all_candidates'] = all_candidates

    if load_filter_conditions:
        all_conditions = np.vstack(all_conditions)
        if all_conditions.shape[1] == 11:
            col_names = [
                'in_bounds',
                'center_close_to_guess_xy',
                'center_close_to_guess_z',
                'xy_size_small_enough',
                'xy_size_big_enough',
                'z_size_small_enough',
                'z_size_big_enough',
                'amp_ok',
                'sigma_ratio_small_enough',
                'sigma_ratio_big_enough',
                'unique',
                ]
        elif all_conditions.shape[1] == 17:
            col_names = [
                'x-position too small',
                'x-position too large',
                'y-position too small',
                'y-position too large',
                'z-position too small',
                'z-position too large',
                'xy-size too small',
                'xy-size too large',
                'z-size too small',
                'z-size too large',
                'amplitude too small',
                'amplitude too large',
                'xy deviation too small',
                'xy deviation too large',
                'z deviation too small',
                'z deviation too large',
                'not unique',
                ]
        all_conditions = pd.DataFrame(data=all_conditions, columns=col_names)
        all_data['all_conditions'] = all_conditions
    
    return all_data

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")