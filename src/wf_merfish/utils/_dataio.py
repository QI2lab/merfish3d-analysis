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
import scipy.sparse as sparse
import scipy.io as sio
import subprocess
import csv

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

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_mtx(baysor_output_path: Path, 
               output_dir_path: Path, 
               confidence_cutoff: float = 0.7, 
               rep_int:int = 100000):

    try:
        baysor_path = Path(baysor_output_path)
        if not baysor_path.exists():
            raise FileNotFoundError(f"The specified Baysor output ({baysor_path}) does not exist!")
    except FileNotFoundError as e:
        print(e)
        return

    # Create a unique directory if the specified directory exists
    if output_dir_path.exists():
        output_dir_path = Path(f"{output_dir_path}/{time_stamp()}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        transcripts_df = pd.read_parquet(baysor_path, columns=["gene", "baysor_cell_id", "assignment_confidence"])
        format = 0  # all genes used in baysor
    except:
        try:
            transcripts_df = pd.read_parquet(baysor_path, columns=["gene_id", "baysor_cell_id"])
            transcripts_df["assignment_confidence"] = 1.0  # no baysor confidence due to clustering on subset
            format = 1  # some genes excluded in baysor
        except:
            transcripts_df = pd.read_csv(baysor_path, usecols=["gene_id", "cell_id"])
            transcripts_df["assignment_confidence"] = 1.0  # no baysor confidence due to clustering on subset
            format = 2  # cellpose segmentations

    features = np.unique(transcripts_df["gene" if format == 0 else "gene_id"])
    feature_to_index = {str(val): index for index, val in enumerate(features)}
    cell_column = "baysor_cell_id" if format in [0, 1] else "cell_id"
    cells = np.unique(transcripts_df[cell_column])
    matrix = pd.DataFrame(0, index=range(len(features)), columns=cells, dtype=np.int32)

    for index, row in transcripts_df.iterrows():
        if index % rep_int == 0:
            print(f"{index} transcripts processed.")

        feature = row['gene' if format == 0 else 'gene_id']
        cell = row[cell_column]
        conf = row['assignment_confidence']
        if conf < confidence_cutoff:
            continue
        if cell:
            matrix.at[feature_to_index[feature], cell] += 1

    write_sparse_mtx(output_dir_path, matrix, cells, features)

def write_sparse_mtx(output_dir_path, matrix, cells, features):

    sparse_mat = sparse.coo_matrix(matrix.values)
    sio.mmwrite(str(output_dir_path / "matrix.mtx"), sparse_mat)
    write_tsv(output_dir_path / "barcodes.tsv", ["cell_" + str(cell) for cell in cells])
    write_tsv(output_dir_path / "features.tsv", [[str(f), str(f), "Blank Codeword" if str(f).startswith("Blank") else "Gene Expression"] for f in features])
    subprocess.run(f"gzip -f {str(output_dir_path)}/*", shell=True)

def write_tsv(filename, data):
    with open(filename, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for item in data:
            writer.writerow([item] if isinstance(item, str) else item)