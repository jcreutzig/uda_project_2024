# Filename: load_data.py
# Author: Jakob Creutzig
# Date: 2024-12-31
# Description: Load data from the source and create a DataLoader object for the data.

import os
import zipfile
from typing import Dict

import h5py
import numpy as np
import torch
from requests import get
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# CONSTANTS
EDGE_WGT = 9.0  # weight for edges.  Not sure why not in dataset.


# FUNCTIONS
def mkdir_if_not_exist(folder):
    """helper funciton to make dir"""
    if not os.path.exists(folder):
        os.makedirs(folder)


def make_path_if_not_exist(file):
    """helper funciton to make whole path"""
    folder = os.path.dirname(file)
    mkdir_if_not_exist(folder)


def load_from_source(url, file, overwrite=False):
    """helper function to download data from source and save"""
    make_path_if_not_exist(file)
    if not os.path.exists(file) or overwrite:
        with open(file, "wb") as f:
            response = get(url, timeout=90)
            f.write(response.content)
            print(f"Downloaded {url} to {file}")
            f.close()

    else:
        print(f"File {file} already exists. Set overwrite=True to overwrite")


def extract_zip(zipfile_name, folder):
    """helper function to extract zip file"""
    zf = zipfile.ZipFile(zipfile_name)
    zf.extractall(folder)
    zf.close()


def load_data(ds: str, purpose: str, data_folder: str, batch_size=500) -> DataLoader:
    """Extract data from h5 file and create DataLoader object.

    This is setup specific and will need modificaiton for other datasets.
    Here we assume we have data input_data.h5 containing 'grids', and the labels
    in 'snbs.h5'.

    """
    folder_name = f"{data_folder}/{ds}/{purpose}/"

    file_name_input = f"{folder_name}/input_data.h5"
    file_name_snbs = f"{folder_name}/snbs.h5"

    snbs_data = h5py.File(file_name_snbs, "r")
    input_data = h5py.File(file_name_input, "r")["grids"]

    grid_data = [
        Data(
            x=tensor(np.array(id_v["node_features"]).reshape(-1, 1), dtype=torch.float),
            edge_index=tensor(np.array(id_v["edge_index"]).T - 1, dtype=torch.long),
            edge_attr=tensor(
                np.full(shape=id_v["edge_index"].shape[0], fill_value=EDGE_WGT),
                dtype=torch.float,
            ),
            y=tensor(np.array(snbs_v), dtype=torch.float),
        )
        for id_v, snbs_v in zip(input_data.values(), snbs_data.values())
    ]

    return DataLoader(grid_data, batch_size=batch_size)


def load_dataset(ds: str, data_folder: str) -> Dict[str, DataLoader]:
    """Convenience function to load train/test/validation data"""
    res = dict()
    res["train_ds"] = load_data(ds=ds, purpose="train", data_folder=data_folder)
    res["test_ds"] = load_data(ds=ds, purpose="test", data_folder=data_folder)
    res["valid_ds"] = load_data(ds=ds, purpose="valid", data_folder=data_folder)
    return res
