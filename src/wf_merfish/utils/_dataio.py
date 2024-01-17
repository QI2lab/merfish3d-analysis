#!/usr/bin/env python

'''
QI2lab OPM suite
Reconstruction tools

Read and write metadata, read raw data, Zarr creation, Zarr conversion
'''

import re
from typing import Dict, Union
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

def load_localizations(
    dir_load, rounds=None, x_pos=None, y_pos=None, z_pos=None, channels=None,
    has_xdata=True, extra_str='', save_x_idx=True, save_y_idx=True, save_z_idx=True, 
    save_r_idx=True, save_ch_idx=True, round_idx_to_name=False, n_ch=None, ch_offset=1,
    load_fitted_var=True, load_candidates=True, load_filter_conditions=True,
    filter_inplace=False, localize_format=1):
    """
    Read files from spot localizations and assemble them.
    
    Parameters
    ----------
    localize_format : int
        Version of file formats and structure used during decoding.
        1: use .parquet file
        0: use .npy files
    """

    if rounds is None:
        # rounds *names* from 1 to end included
        rounds = range(8)
    if x_pos is None:
        x_pos = [0]
    if y_pos is None:
        y_pos = [0]
    if z_pos is None:
        z_pos = [0]
    if channels is None:
        channels = [1, 2]
    if n_ch is None:
        n_ch = len(channels)
    if localize_format == 1:
        fext = 'parquet'
        read_fct = pd.read_parquet
    elif localize_format == 0:
        fext = 'npy'
        read_fct = np.load

    all_spots = []
    all_bits = []
    all_fit_vars = []
    all_candidates = []
    all_conditions = []
    all_x_idx = []
    all_y_idx = []
    all_z_idx = []
    all_r_idx = []
    all_ch_idx = []
    
    for r_idx in rounds:
        if round_idx_to_name:
            r_name = r_idx + 1
        else:
            r_name = r_idx
        for ch_idx in channels:
            bit_idx = get_bit_from_round(r_idx, ch_idx, n_ch, ch_offset)
            for x_idx in x_pos:
                for y_idx in y_pos:
                    for z_idx in z_pos:
                        if has_xdata:
                            base_name = f'r00{r_name:0>2d}_x00{x_idx:0>2d}_y00{y_idx:0>2d}_z00{z_idx:0>2d}_ch_idx-{ch_idx}{extra_str}'
                        else:
                            base_name = f'r00{r_name:0>2d}_y00{y_idx:0>2d}_z00{z_idx:0>2d}_ch_idx-{ch_idx}{extra_str}'
                        # load coords z / y / x and boolean selector
                        file_name = f'localized_spots_{base_name}.{fext}'
                        spots = read_fct(dir_load / file_name)
                        # use only selected spots after filtering
                        if filter_inplace:
                            if localize_format == 1:
                                select = spots['select'].values
                                spots = spots.loc[select, ['z', 'y', 'x']].values
                            elif localize_format == 0:
                                select = spots[:, -1].astype(bool)
                                spots = spots[select, :3]
                        all_spots.append(spots)

                        # keep track of bit id across rounds and channels
                        all_bits.extend([bit_idx] * len(spots))
                        if save_r_idx:
                            all_r_idx.extend([r_idx] * len(spots))
                        if save_ch_idx:
                            all_ch_idx.extend([ch_idx] * len(spots))
                        if save_x_idx:
                            all_x_idx.extend([x_idx] * len(spots))
                        if save_y_idx:
                            all_y_idx.extend([y_idx] * len(spots))
                        if save_z_idx:
                            all_z_idx.extend([z_idx] * len(spots))

                        # load fitted variables
                        if load_fitted_var:                    
                            file_name = f'fitted_variables_{base_name}.{fext}'
                            fit_vars = read_fct(dir_load / file_name)
                            if filter_inplace:
                                if localize_format == 1:
                                    fit_vars = fit_vars.loc[select, :]
                                elif localize_format == 0:
                                    fit_vars = fit_vars[select, :]
                            all_fit_vars.append(fit_vars)

                        # load candidates
                        if load_candidates:                    
                            file_name = f'localization_candidates_{base_name}.{fext}'
                            candidates = read_fct(dir_load / file_name)
                            if filter_inplace:
                                if localize_format == 1:
                                    candidates = candidates.loc[select, :]
                                elif localize_format == 0:
                                    candidates = candidates[select, :]
                            all_candidates.append(candidates)

                        # load filter conditions
                        if load_filter_conditions:                    
                            file_name = f'filter_conditions_{base_name}.{fext}'
                            conditions = read_fct(dir_load / file_name)
                            if filter_inplace:
                                if localize_format == 1:
                                    conditions = conditions.loc[select, :]
                                elif localize_format == 0:
                                    conditions = conditions[select, :]
                            all_conditions.append(conditions)

    
    all_spots = np.vstack(all_spots)
    col_names = ['z', 'y', 'x']
    if not filter_inplace:
        # boolean selector was not discarded
        # /!\ this assumes `select` is the last column /!\
        col_names = col_names + ['select']
    detected_coords = pd.DataFrame(data=all_spots, columns=col_names)
    detected_coords['bit_idx'] = all_bits
    if save_r_idx:
        detected_coords['rounds'] = all_r_idx
    if save_ch_idx:
        detected_coords['channels'] = all_ch_idx
    if save_x_idx:
        detected_coords['x_idx'] = all_x_idx
    if save_y_idx:
        detected_coords['y_idx'] = all_y_idx
    if save_z_idx:
        detected_coords['z_idx'] = all_z_idx
    
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