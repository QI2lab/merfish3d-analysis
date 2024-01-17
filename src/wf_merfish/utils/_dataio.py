#!/usr/bin/env python

import re
import pandas as pd
from datetime import datetime
import os
import pathlib

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

def append_index_filepath(filepath):
    """
    Append a number to a file path if the file already exists,
    the number increases as long as there is a file that exists.
    """
    if isinstance(filepath, (pathlib.WindowsPath, pathlib.PosixPath)):
        to_pathlib = True
        filepath = str(filepath.as_posix())
    else:
        to_pathlib = False

    i = 1
    while os.path.exists(filepath):
        filepath = "".join(filepath.split('.')[:-1]) + f"-{i}." + filepath.split('.')[-1]
        i += 1
    if to_pathlib:
        filepath = pathlib.Path(filepath)
    return filepath

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

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")